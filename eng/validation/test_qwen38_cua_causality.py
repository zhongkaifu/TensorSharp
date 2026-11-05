"""CPU contracts only. These mocks do not run models, generated code or CUDA."""
import ast,contextlib,copy,hashlib,importlib.util,io,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch,Mock
HERE=Path(__file__).resolve().parent

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
M=load(HERE/'qwen38-cua-causality.py','test_causal_controller')
B=load(HERE/'qwen38_vm_guard.py','test_causal_vm_guard')
A=load(HERE/'qwen38_cua_guard.py','test_causal_cua_guard')

def config():
 return {'schema':'qwen38-cua-causality-jobs/1','profiles':[{'name':'ScalarA1','cpu_simd':'0','splitk_policy':'production-unset'},{'name':'ScalarA2','cpu_simd':'0','splitk_policy':'production-unset'}],'comparisons':[{'left':'ScalarA1','right':'ScalarA2'}]}
def envargs():
 return types.SimpleNamespace(source=Path('/dev/shm/source'),runtime=Path('/dev/shm/runtime'),work=Path('/dev/shm/fresh'),models=Path('/dev/shm/models'),tokenizer=Path('/workspace/tokenizer'),haystack=Path('/workspace/haystack'),model_manifest=Path('/workspace/models.json'),bench_python=Path('/root/venvs/bench/bin/python'),port=5088,startup_timeout=330)

class Contracts(unittest.TestCase):
 def test_tracked_helper_pins_and_literal_frozen_function_AST(self):
  for module,pin,name in((B,M.MATRIX_SHA,'qwen38_vm_guard.py'),(A,M.AUDIT_SHA,'qwen38_cua_guard.py')):
   self.assertEqual(M.sha(HERE/name),pin);tree=ast.parse((HERE/name).read_text());actual={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()for n in tree.body if isinstance(n,ast.FunctionDef)};self.assertEqual(actual,module.FROZEN_FUNCTION_AST_SHA256)
   self.assertFalse(any(isinstance(n,ast.FunctionDef)and n.name=='main'for n in tree.body));self.assertNotIn('root-media-guard',(HERE/name).read_text());self.assertNotIn('qwen38-ignis-e2e-matrix',(HERE/name).read_text())
  self.assertEqual(B.ORIGIN_SHA256,'b8872d490db0e448a2ac00f9ffbebd3281d2c2c61274f258120436e17ee75965');self.assertEqual(A.ORIGIN_SHA256,'bcb1601a5cca35c96d4ed4fc3d427e9fe73386d4863809df8cf9037faf8f7c4a')
 def test_original_quality_measurement_and_producer_pins(self):
  self.assertEqual(A.QUALITY,{'http_success','single_choice','stream_complete','no_hidden_thinking','one_correct_tool','typed_exact_arguments','no_freeform_answer'});self.assertEqual(A.MEASUREMENT,{'actual_server_usage','finite_client_timings','ttft_within_request'});self.assertEqual(A.CUA_SHA,'ab25261e8f3bcdc4ed2dfee8c010d6802b57cd6fb29a881daf63536a344eaf69');self.assertEqual(B.PINS['eng/qwen38-engine-bench.py'],'4e1fa314732250648f43a06d62c02d82b9c17e8842edfeda812f969aac8c363c')
 def test_full_AA_denominator_and_directed_order(self):
  c=M.configurations(config());jobs=M.jobs(c,'both');self.assertEqual(len(jobs),4);self.assertEqual(sum(j['expected_requests']for j in jobs),384);self.assertEqual([j['configuration']for j in jobs],['ScalarA1','ScalarA2','ScalarA1','ScalarA2']);self.assertEqual(sum(j['expected_requests']for j in M.jobs(c,'text')),360);self.assertEqual(sum(j['expected_requests']for j in M.jobs(c,'media')),24)
 def test_invalid_profiles_pairs_and_simultaneous_policy_changes_rejected(self):
  for mode in('duplicate','extra_env','cpu_bool','split','unbound','self','pair_duplicate','bothchanges'):
   c=config()
   if mode=='duplicate':c['profiles'][1]['name']='ScalarA1'
   elif mode=='extra_env':c['profiles'][0]['env']={'TS_Q35_SHARED_SCRATCH':'1'}
   elif mode=='cpu_bool':c['profiles'][0]['cpu_simd']=False
   elif mode=='split':c['profiles'][0]['splitk_policy']='auto'
   elif mode=='unbound':c['comparisons'][0]['right']='unknown'
   elif mode=='self':c['comparisons'][0]['right']='ScalarA1'
   elif mode=='pair_duplicate':c['comparisons'].append(c['comparisons'][0])
   else:c['profiles'][1].update(cpu_simd='1',splitk_policy='splitk-zero')
   with self.subTest(mode=mode),self.assertRaises(ValueError):M.configurations(c)
 def test_environment_M0_and_explicit_SPLITK_profile(self):
  a=envargs();job=M.jobs(M.configurations(config()),'both')[0]
  for policy in M.SPLITS:
   j=dict(job,splitk_policy=policy);env=M.environment(B,a,j,Path('/dev/shm/fresh/job'));self.assertTrue(all(env[k]=='0'for k in B.ALT));self.assertEqual(env['TS_TOKEN_SAMPLER_SIMD'],'0');self.assertEqual(env['TS_Q35_SKINNY_FFN_A4'],'0');self.assertNotIn('TS_Q35_MIXED_FP4_VERIFY',env);self.assertIn('TS_Q35_MIXED_FP4_VERIFY',env['MANIFEST_ENV_ALLOWLIST'].split(','))
   for key in M.SPLIT_ENV:
    self.assertIn(key,env['MANIFEST_ENV_ALLOWLIST'].split(','))
    if policy=='production-unset':self.assertNotIn(key,env)
    else:self.assertEqual(env[key],'0')
 def test_actual_manifest_split_policy_expression_rejects_missing_or_other_policy(self):
  tree=ast.parse((HERE/'qwen38-cua-causality.py').read_text());nodes=[n for n in ast.walk(tree)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='require'and len(n.value.args)>1 and isinstance(n.value.args[1],ast.Constant)and n.value.args[1].value=='actual cap/context or explicitly selected splitK math policy differs'];self.assertEqual(len(nodes),1);code=compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual_split_gate','exec')
  for policy in M.SPLITS:
   actual={'MAX_CONTEXT':'65536','TS_SCHED_MAX_RUNNING_SEQS':'2',**{k:'0'if policy=='splitk-zero'else None for k in M.SPLIT_ENV}};ns=dict(require=M.require,actual=actual,m={'sequence_admission_configuration':{'configured_running_sequence_limit':'2'}},job={'splitk_policy':policy},SPLIT_ENV=M.SPLIT_ENV);exec(code,ns)
   for key in M.SPLIT_ENV:
    bad=dict(actual);bad.pop(key)
    with self.assertRaises(ValueError):exec(code,dict(ns,actual=bad))
    bad=dict(actual);bad[key]=None if policy=='splitk-zero'else'0'
    with self.assertRaises(ValueError):exec(code,dict(ns,actual=bad))
 def test_PLAN_only_no_process_or_GPU_calls(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'jobs.json';p.write_text(json.dumps(config()))
   with patch.object(M.subprocess,'Popen',side_effect=AssertionError('PLAN spawned')),patch.object(M.subprocess,'check_output',side_effect=AssertionError('PLAN GPUquery')),contextlib.redirect_stdout(io.StringIO())as out:
    self.assertEqual(M.main(['--source','/dev/shm/source','--runtime','/dev/shm/runtime','--build-campaign','/dev/shm/control/campaign.json','--jobs-json',str(p)]),0)
   state=json.loads(out.getvalue());self.assertTrue(state['plan_only']);self.assertFalse(state['complete']);self.assertFalse(state['quality_passed']);self.assertEqual(sum(x['expected_requests']for x in state['jobs']),384)
 def test_safe_producer_path_and_alias_rejection(self):
  cua=load(HERE/'qwen38-cua-control.py','frozen_path_validation_only')
  with tempfile.TemporaryDirectory(dir=HERE.parents[1]/'artifacts')as t:
   root=Path(t).resolve();h=root/'harness';h.mkdir();work=Path('/dev/shm/causal-test/fresh');p=M.replay_path(B,cua,h,work,'ScalarA1-text');self.assertEqual(cua.evidence_path(str(p)),p);self.assertFalse(p.exists())
   (h/'artifacts').symlink_to(root,target_is_directory=True)
   with self.assertRaises(ValueError):M.replay_path(B,cua,h,work,'ScalarA1-text')
 def test_foreign_record_does_not_stop(self):
  with tempfile.TemporaryDirectory()as t:
   record=Path(t)/'record.json';manifest=Path(t)/'manifest.json';record.write_text(json.dumps({'manifest':'foreign'}));command=Mock(side_effect=AssertionError('foreign stop'))
   with self.assertRaises(ValueError):M.cleanup_owned(B,types.SimpleNamespace(harness=Path('/h')),{},record,manifest,True,{},command)
   self.assertTrue(record.exists());command.assert_not_called()
 def test_same_record_cleanup_uses_only_owned_record(self):
  with tempfile.TemporaryDirectory()as t:
   record=Path(t)/'record.json';manifest=Path(t)/'manifest.json';owner={'manifest':str(manifest),'pid':12};record.write_text(json.dumps(owner));attempt={'process_record':{'value':owner}}
   def stop(attempt,name,argv,timeout,env):
    self.assertEqual(name,'cleanup');self.assertIn('--record',argv);self.assertEqual(argv[argv.index('--record')+1],record);record.unlink();return {'complete':True}
   M.cleanup_owned(B,types.SimpleNamespace(harness=Path('/h')),attempt,record,manifest,True,{},stop);self.assertFalse(record.exists())
 def test_pair_keeps_original_and_typed_regressions_without_netting(self):
  with tempfile.TemporaryDirectory()as t:
   paths=[Path(t)/x for x in('left','right')]
   def row(code,good,other):return dict(kind='request',scenario='append',lanes=2,lane=1,step=3,expected={'code':9970},common_sha256='a',canonical_input_sha256='b',full_body_sha256='c',wire_request_body_sha256='d',quality={'checks':{'typed_exact_arguments':good,'other':other},'measurement_checks':{'actual_server_usage':True}},response={'response_sha256':'same','tool_calls':[{'name':'submit_observation','parsed':{'code':code}}]},raw_transport={'body_sha256':'same'})
   paths[0].write_text(json.dumps(row(9970,True,False))+'\n');paths[1].write_text(json.dumps(row(9976,False,True))+'\n');p=M.pair_fields(A,*paths);self.assertEqual(p['original_check_true_to_false'],1);self.assertEqual(p['typed_field_true_to_false'],1);self.assertEqual(p['observations'][0]['original_checks_false_to_true'],['other']);self.assertFalse(p['quality_equivalence_claim'])
 def test_S4_build_pins_fail_closed(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'build';q=Path(t)/'catalog';p.write_text('{}');q.write_text('[]')
   with self.assertRaises(ValueError):M.verify_S4({'complete':True,'new_runtime':[]},[],p,q)

if __name__=='__main__':unittest.main()
