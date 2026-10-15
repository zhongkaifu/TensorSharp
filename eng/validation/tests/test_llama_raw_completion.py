"""Raw-token conditioning, strict quality and owned deadline CPU fixtures."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('llama_raw', Path(__file__).parents[1] / 'llama-raw-completion.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture():
    body = dict(prompt=[11, 22], n_predict=8, temperature=0.0, samplers=['temperature'],
        top_k=0, top_p=1.0, min_p=0.0, repeat_penalty=1.0, repeat_last_n=0,
        presence_penalty=0.0, frequency_penalty=0.0, dry_multiplier=0.0,
        cache_prompt=False, return_tokens=True, n_probs=0, stream=False, ignore_eos=False, stop=[])
    plan = dict(futureCommand=['server', '--host', '127.0.0.1', '--port', '52178',
        '--parallel', '1', '--ctx-size', '32', '--spec-type', 'none'],
        futureRequest=dict(endpoint='http://127.0.0.1:52178/completion', body=body),
        prompt=dict(tokens=[11, 22], count=2), expected_checkpoint_eos_ids=[99])
    return plan, body


def response():
    return dict(content='\n {"name":"get_weather","arguments":{"city":"Hangzhou","unit":"celsius"}}\n',
                tokens=[71, 72, 99], tokens_evaluated=2, tokens_predicted=3, stop_type='eos', truncated=False)


class RawPlanTests(unittest.TestCase):
    def test_exact_integer_prompt_kept(self):
        plan, body = fixture()
        before = copy.deepcopy(body)
        self.assertEqual('/completion', runner.validate_plan(plan, body).path)
        self.assertEqual(before, body)

    def test_chat_or_nonlocal_endpoint_rejected(self):
        for url in ('http://127.0.0.1:52178/v1/chat/completions', 'http://example.org:52178/completion',
                    'http://127.0.0.1:52178/completion?q=x', 'http://u:p@127.0.0.1:52178/completion'):
            with self.subTest(url=url):
                plan, body = fixture(); plan['futureRequest']['endpoint'] = url
                with self.assertRaises(ValueError): runner.validate_plan(plan, body)

    def test_prompt_mutation_string_and_boolean_are_rejected(self):
        for prompt in ([11, 23], 'same rendered words', [True, 22], [11, -1], []):
            with self.subTest(prompt=prompt):
                plan, body = fixture(); body['prompt'] = prompt
                with self.assertRaises(ValueError): runner.validate_plan(plan, body)

    def test_request_cannot_add_formatting_constraints(self):
        for key, value in [('grammar', 'root ::= "42"'), ('json_schema', {}), ('response_format', {}), ('tools', [])]:
            with self.subTest(key=key):
                plan, body = fixture(); body[key] = value
                with self.assertRaises(ValueError): runner.validate_plan(plan, body)

    def test_sampling_stop_and_bound_controls_must_be_explicit(self):
        for key, value in [('stream', True), ('ignore_eos', True), ('return_tokens', False),
                           ('stop', ['\n']), ('temperature', .5), ('n_predict', True), ('n_predict', 0),
                           ('n_predict', 100), ('repeat_penalty', 1.1), ('cache_prompt', True), ('temperature', False)]:
            with self.subTest(key=key):
                plan, body = fixture(); body[key] = value
                with self.assertRaises(ValueError): runner.validate_plan(plan, body)

    def test_server_must_be_single_slot_no_draft_and_same_port(self):
        for option, value in [('--port', '52179'), ('--parallel', '2'), ('--spec-type', 'mtp')]:
            plan, body = fixture(); plan['futureCommand'][plan['futureCommand'].index(option) + 1] = value
            with self.subTest(option=option), self.assertRaises(ValueError): runner.validate_plan(plan, body)


class RawQualityTests(unittest.TestCase):
    def test_free_text_natural_eos_is_record_only_not_semantic_pass(self):
        _, body = fixture(); answer = response(); answer['content'] = 'An unverified factual statement.'
        result = runner.quality_result(answer, body, [99], 'free_text')
        self.assertTrue(result['passed']); self.assertTrue(result['record_only'])
        self.assertIsNone(result['semantic_pass']); self.assertIsNone(result['semantic_check']['passed'])
        self.assertTrue(result['natural_eos']); self.assertFalse(result['bounded_limit'])
        self.assertEqual(answer['content'], result['content'])

    def test_free_text_exact_cap_is_valid_record_not_complete_answer(self):
        _, body = fixture(); answer = response()
        answer.update(tokens=[71]*8, tokens_predicted=8, stop_type='limit')
        result = runner.quality_result(answer, body, [99], 'free_text')
        self.assertTrue(result['passed']); self.assertTrue(result['bounded_limit'])
        self.assertFalse(result['natural_eos']); self.assertIsNone(result['semantic_pass'])
        for key, value in [('tokens',[71]*7), ('tokens',[71]*7+[99]), ('truncated',True),
                           ('stop_type','word'), ('tokens_evaluated',1), ('tokens',[2**31]*8)]:
            bad=copy.deepcopy(answer); bad[key]=value
            with self.subTest(key=key,value=value):
                self.assertFalse(runner.quality_result(bad,body,[99],'free_text')['passed'])

    def test_recorded_trajectory_divergence_is_descriptive(self):
        row=runner.compare_trajectory([1,2,3], {'tokens':[1,4,99]})
        self.assertEqual(1,row['first_mismatch_index']); self.assertFalse(row['full_trajectory_exact'])
        self.assertNotIn('semantic_pass',row)
        row=runner.compare_trajectory([1,2], {'tokens':[1,2,99]})
        self.assertTrue(row['compared_prefix_exact']); self.assertFalse(row['full_trajectory_exact'])

    def test_int32_token_hash_is_little_endian_and_rejects_coercion(self):
        import hashlib,struct
        self.assertEqual(hashlib.sha256(struct.pack('<ii',1,256)).hexdigest(),runner.token_hash([1,256]))
        for values in ([],[True],[-1],[2**31],[float('nan')]):
            with self.subTest(values=values),self.assertRaises(ValueError): runner.token_hash(values)

    def test_raw_json_with_whitespace_and_real_eos_passes(self):
        _, body = fixture()
        result = runner.quality_result(response(), body, [99], 'tool_json')
        self.assertTrue(result['passed'])
        self.assertEqual(response()['content'], result['content'])

    def test_fences_duplicate_or_extra_keys_fail_without_repair(self):
        for text in ('```json\n' + response()['content'] + '```',
                     '{"name":"get_weather","name":"get_weather","arguments":{"city":"Hangzhou","unit":"celsius"}}',
                     '{"name":"get_weather","arguments":{"city":"Hangzhou","unit":"celsius"},"extra":1}'):
            with self.subTest(text=text):
                _, body = fixture(); answer = response(); answer['content'] = text
                result = runner.quality_result(answer, body, [99], 'tool_json')
                self.assertFalse(result['passed']); self.assertTrue(result['natural_eos'])
                self.assertEqual(text, result['content'])

    def test_token_truncation_missing_eos_early_eos_or_count_mismatch_fails(self):
        cases = [('tokens', [71,72]), ('tokens', [99,71,99]), ('tokens', [True,99]),
                 ('tokens', []), ('tokens', [71]*9+[99]), ('tokens', None),
                 ('tokens_evaluated', 3), ('tokens_predicted', 2), ('tokens_predicted', None),
                 ('stop_type', 'limit'), ('truncated', True)]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                _, body = fixture(); answer = response(); answer[key] = value
                self.assertFalse(runner.quality_result(answer, body, [99], 'tool_json')['passed'])

    def test_missing_content_is_recorded_not_repaired(self):
        _, body = fixture(); answer = response(); answer.pop('content')
        self.assertFalse(runner.quality_result(answer, body, [99], 'tool_json')['passed'])


class RawLifetimeTests(unittest.TestCase):
    def test_checker_cannot_relabel_reviewed_prompt(self):
        plan, body = fixture()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'Semantic case'):
            runner.run(plan, body, b'{}', Path(tmp)/'run', 5, 'math')

    def test_watchdog_kills_and_cleanup_reaps_real_owned_cpu_child(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                 creationflags=runner.bench.PROCESS_FLAGS)
        expired = threading.Event(); errors = []
        timer = runner.start_watchdog(child, .05, expired, errors)
        try:
            child.wait(timeout=10)
            self.assertTrue(expired.is_set()); self.assertFalse(errors)
        finally:
            cleanup = runner.owned.cleanup_owned(child, timer, None)
        self.assertTrue(cleanup['owned_server_exited']); self.assertFalse(cleanup['cleanup_errors'])

    def test_preflight_source_failure_leaves_terminal_evidence_without_server(self):
        plan, body = fixture(); plan['source'] = dict(path='missing', revision='abc')
        plan['deployment'] = dict(files={})
        # Ephemeral unused port; no HTTP listener is created by this fixture.
        import socket
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',0)); port=probe.getsockname()[1]
        plan['futureCommand'][plan['futureCommand'].index('--port')+1]=str(port)
        plan['futureRequest']['endpoint']=f'http://127.0.0.1:{port}/completion'
        import json
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner.subprocess, 'check_output', side_effect=ValueError('source failure')):
            out=Path(tmp)/'evidence'
            result=runner.run(plan,body,json.dumps(body).encode(),out,5,'tool_json')
            self.assertFalse(result['passed']); self.assertTrue(result['owned_server_exited'])
            self.assertNotIn('server_pid',result); self.assertTrue((out/'execution.json').exists())
            self.assertIn('source failure',result['error'])


class LoadedIdentityTests(unittest.TestCase):
    def test_free_text_conditioning_is_bound_to_actual_export(self):
        import json
        plan,_=fixture(); plan.update(case='free_text',record_only=True,semantic_pass=None)
        plan['prompt']['i32le_sha256']=runner.token_hash(plan['prompt']['tokens'])
        plan['checkpoint']={'shards':[{'path':'checkpoint','sha256':'a'*64}]}
        plan['reference_output']={'tokens':[71,99], 'i32le_sha256':runner.token_hash([71,99])}
        export={'status':'extracted-not-baseline-executed','requests':[{'http_request_id':'http-1',
            'prompt':plan['prompt'],'output':plan['reference_output']}], 'identity':{'checkpoint':plan['checkpoint']}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tokens.json'; path.write_text(json.dumps(export),encoding='utf-8')
            plan['conditioning_evidence']={'path':str(path),'sha256':runner.bench.sha(path),'request_index':0,'http_request_id':'http-1'}
            self.assertEqual('http-1',runner.conditioning_evidence(plan)['http_request_id'])
            for key,value in [('request_index',True),('http_request_id','other'),('sha256','0'*64)]:
                bad=copy.deepcopy(plan); bad['conditioning_evidence'][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError): runner.conditioning_evidence(bad)
            plan['reference_output']['tokens'][0]=72
            with self.assertRaises(ValueError): runner.conditioning_evidence(plan)

    def test_required_live_modules_must_match_pinned_paths_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[Path(tmp)/name for name in ('llama-server.exe','llama.dll','ggml-cuda.dll')]
            for p in paths:p.write_bytes(b'pinned')
            plan={'futureCommand':[str(paths[0])],'deployment':{'files':{p.name:{'path':str(p),'sha256':runner.bench.sha(p)} for p in paths}}}
            output='\n'.join(str(p) for p in paths)
            with mock.patch.object(runner.sys,'platform','win32'), mock.patch.object(runner.subprocess,'run',return_value=mock.Mock(stdout=output)):
                seen=runner.observe_modules(123,plan,10)
            self.assertEqual({p.name for p in paths},set(seen))
            paths[-1].write_bytes(b'changed')
            with mock.patch.object(runner.sys,'platform','win32'), mock.patch.object(runner.subprocess,'run',return_value=mock.Mock(stdout=output)), self.assertRaisesRegex(ValueError,'digest differs'):
                runner.observe_modules(123,plan,10)

    def test_missing_or_duplicate_loaded_cuda_module_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[Path(tmp)/name for name in ('llama-server.exe','llama.dll','ggml-cuda.dll')]
            for p in paths:p.write_bytes(b'pinned')
            plan={'futureCommand':[str(paths[0])],'deployment':{'files':{p.name:{'path':str(p),'sha256':runner.bench.sha(p)} for p in paths}}}
            for loaded in (paths[:-1],paths+[paths[-1]]):
                with self.subTest(loaded=loaded), mock.patch.object(runner.sys,'platform','win32'), mock.patch.object(runner.subprocess,'run',return_value=mock.Mock(stdout='\n'.join(map(str,loaded)))), self.assertRaises(ValueError):
                    runner.observe_modules(123,plan,10)


if __name__ == '__main__':
    unittest.main()
