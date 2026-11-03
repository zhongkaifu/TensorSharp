import copy,importlib.util,json,math,os,signal,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
HERE=Path(__file__).resolve().parent
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
B=load(HERE/'build.py','schedule_cpu_build');R=load(HERE/'run.py','schedule_cpu_run')
def fixture(bench=True,candidates=None):
    c=B.CANDIDATES if candidates is None else candidates;m=dict(arm_selection='all'if len(c)==3 else('bounds','stream','combo')[B.CANDIDATES.index(c[0])],candidates=c)
    events=[dict(event='device',name='CPU MOCK NOT A DEVICE',sms=170,registers_per_sm=65536,driver=13020,runtime=13020)]
    for symbol in [B.ORACLE,*(v['symbol']for v in c)]:events.append(dict(event='driver_resources',symbol=symbol,rows_per_warp=4,block_threads=128,registers=168,local_bytes=0,static_shared_bytes=0,max_block_threads=128,active_ctas=3,binary_version=120))
    events += [dict(event='production_rows_query',actual_rows_per_warp=4,expected_rows_per_warp=4,guards=True),dict(event='driver_launch_gate',launchable=True,full_quality_plan_passed=False)]
    for v in c:
        for s in B.SEALED.plan():
            events.append(dict(event='strict_case_pass',symbol=v['symbol'],shape=R.name(s),rows_per_warp=4,block_threads=128,k=s['k'],out_dim=s['out_dim'],pattern=s['pattern'],alpha_bits=s['alpha_bits'],full_output_bits=True,oracle_AA=True,candidate_AA=True,guards=True,actual_output_argmax_equal=True,outputs=8*s['out_dim'],actual_output_argmax=[0]*8))
            if bench and s['timed']:
                for round in range(5):events.append(dict(event='ABBA_gpu_ms',symbol=v['symbol'],shape=R.name(s),round=round,rows_per_warp=4,block_threads=128,k=s['k'],out_dim=s['out_dim'],role=v['role'],iterations_per_sample=3,fullbits_each_half=True,A1_oracle_ms=1.,B1_candidate_ms=1.,B2_candidate_ms=1.,A2_oracle_ms=1.))
    result=dict(complete=True,status='STRICT_OUTPUT_BITS_AND_ARGMAX_PASS',activation_rows=8,strict_cases_per_candidate=96,strict_candidate_comparisons=96*len(c),candidates=len(c),timed_shapes_per_candidate=12 if bench else 0,rounds=5,iterations_per_sample=3)
    return events,result,m
class Gates(unittest.TestCase):
    def test_original_template_and_cache_delta_reverse(self):
        source=B.SOURCE.read_text();generated=B.generate(source);prefix=B.SEALED.oracle_prefix(source)
        self.assertTrue(generated.startswith('#include <cuda_runtime.h>\n'+prefix))
        self.assertEqual(generated.count(B.NEW_VECTOR),1);self.assertEqual(generated.count(B.NEW_SCALE),1)
        self.assertEqual(generated.count('__launch_bounds__(128,3)'),2)
        self.assertIn('const int4 a = *(const int4*)(xq +',generated)
        for c in B.CANDIDATES:self.assertIn(c['symbol']+'(',generated)
    def test_runner_has_only_reversible_metadata_delta(self):
        old=(B.OLD/'bench.cu').read_text();now=(HERE/'bench.cu').read_text()
        array='''    std::array<Variant,4> candidates{{{1,128,"ts_bench_dp4a_rows_8_r1"},
        {2,128,"ts_bench_dp4a_rows_8_r2"},{3,128,"ts_bench_dp4a_rows_8_r3"},
        {4,32,"ts_bench_dp4a_rows_8_r4"}}};'''
        now=now.replace('    #include "variants.h"',array,1).replace('DP4a R4 register/cache scheduling driver runner.','DP4a rows-per-warp driver runner.',1).replace('usage: dp4a-schedule-bench','usage: dp4a-rows-bench',1)
        now=now.replace('Quote(std::string(v.symbol)=="ts_bench_dp4a_r4_bounds3"?"bounds3_only":std::string(v.symbol)=="ts_bench_dp4a_r4_stream"?"weights_stream_only":"bounds3_and_weights_stream")','Quote(v.rows==4?"launch_only_control":"rows_only_experiment")',1)
        now=now.replace('<<",\\"candidates\\":"<<module.candidates.size()<<",\\"timed_shapes_per_candidate\\":','<<",\\"candidates\\":4,\\"timed_shapes_per_candidate\\":',1)
        self.assertEqual(now,old)
    def test_arithmetic_change_refused(self):
        changed=B.SOURCE.read_text().replace('acc[n][j] += ds[j] * xs * (float)sumi;','acc[n][j] += xs * (ds[j] * (float)sumi);',1)
        with self.assertRaises(ValueError):B.generate(changed)
    def test_resource_spill_stack_shared_and_regs_rejected(self):
        log='\n'.join("ptxas info : Compiling entry function '%s' for 'sm_120a'\nptxas info : 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\nptxas info : Used 168 registers"%s for s in B.SYMBOLS)
        self.assertTrue(all(r['offline_resource_qualified']for r in B.parse_resources(log).values()))
        for old,new in [('0 bytes stack','8 bytes stack'),('0 bytes spill stores','4 bytes spill stores'),('0 bytes spill loads','4 bytes spill loads'),('Used 168 registers','Used 256 registers'),('Used 168 registers','Used 168 registers, 8 bytes smem')]:
            self.assertFalse(all(r['offline_resource_qualified']for r in B.parse_resources(log.replace(old,new,1)).values()))
    def test_full_mock_matrix_is_exact_and_not_a_GPU_claim(self):
        events,result,m=fixture();out=R.validate_observation(events,result,m,True,5,3);self.assertEqual(out['strict_comparisons'],288);self.assertEqual(out['ABBA_events'],180)
    def test_missing_duplicate_strict_and_false_guard_refused(self):
        events,result,m=fixture();i=next(i for i,r in enumerate(events)if r['event']=='strict_case_pass')
        for rows in [events[:i]+events[i+1:],events+[events[i]],copy.deepcopy(events)]:
            if len(rows)==len(events):rows[i]['guards']=False
            with self.assertRaises(ValueError):R.validate_observation(rows,result,m,True,5,3)
    def test_resources_and_fake_summary_cannot_pass(self):
        events,result,m=fixture()
        for local,active in [(4,3),(0,0)]:
            rows=copy.deepcopy(events);rows[1]['local_bytes']=local;rows[1]['active_ctas']=active
            with self.assertRaises(ValueError):R.validate_observation(rows,result,m,True,5,3)
        with self.assertRaises(ValueError):R.validate_observation([],result,m,True,5,3)
    def test_boolean_resource_is_not_actual_integer_observation(self):
        events,result,m=fixture();events[1]['active_ctas']=True
        with self.assertRaises(ValueError):R.validate_observation(events,result,m,True,5,3)
    def test_timing_missing_duplicate_nan_and_wrong_role_refused(self):
        events,result,m=fixture();i=next(i for i,r in enumerate(events)if r['event']=='ABBA_gpu_ms')
        for mode in range(4):
            rows=copy.deepcopy(events)
            if mode==0:rows.pop(i)
            elif mode==1:rows.append(rows[i])
            elif mode==2:rows[i]['B2_candidate_ms']=float('nan')
            else:rows[i]['role']='rows_only_experiment'
            with self.assertRaises(ValueError):R.validate_observation(rows,result,m,True,5,3)
    def test_explicit_single_arm_is_not_automatic_pruning(self):
        for c in B.CANDIDATES:
            events,result,m=fixture(candidates=[c]);out=R.validate_observation(events,result,m,True,5,3);self.assertEqual(out['strict_comparisons'],96);self.assertEqual(out['ABBA_events'],60)
        with self.assertRaises(ValueError):R.selected(dict(arm_selection='all',candidates=[B.CANDIDATES[0]]))
    def test_generate_only_fresh_and_no_permission(self):
        with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts')as tmp:
            target=Path(tmp)/'new'
            with patch.dict(os.environ,{'NVCC_PREPEND_FLAGS':'','NVCC_APPEND_FLAGS':''}):self.assertEqual(B.main(['--out',str(target),'--generate-only']),0)
            m=json.loads((target/'manifest.json').read_text());self.assertFalse(m['complete']);self.assertEqual(m['planned_comparisons'],288);self.assertFalse((target/B.PERMIT).exists())
            with self.assertRaises(FileNotFoundError):R.verify_build(target)
    def test_implicit_flags_refused_before_output(self):
        with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts')as tmp:
            target=Path(tmp)/'new'
            with patch.dict(os.environ,{'NVCC_PREPEND_FLAGS':'--use_fast_math'}):
                with self.assertRaises(SystemExit):B.main(['--out',str(target),'--generate-only'])
            self.assertFalse(target.exists())
    def test_default_runner_is_no_gpu_plan(self):
        with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts')as tmp:
            root=Path(tmp)
            self.assertEqual(R.main(['--build',str(root),'--name','plan']),0)
            state=json.loads((root/'plan/run.json').read_text());self.assertFalse(state['gpu_invoked']);self.assertFalse(state['complete'])
    def test_interruption_targets_only_owned_unreaped_group(self):
        proc=Mock(pid=27182,returncode=None);proc.communicate.side_effect=[KeyboardInterrupt(),('',None)]
        with patch.object(B.subprocess,'Popen',return_value=proc),patch.object(B.os,'getpgid',return_value=27182),patch.object(B.os,'killpg')as kill:
            with self.assertRaises(KeyboardInterrupt):B.command(['unused-cpu-mock'],10)
            kill.assert_called_once_with(27182,signal.SIGTERM)
    def test_registration_save_error_also_cleans_own_group(self):
        proc=Mock(pid=27183,returncode=None);proc.communicate.return_value=('',None)
        def failed_save(process):raise OSError('mock registration save failure')
        with patch.object(B.subprocess,'Popen',return_value=proc),patch.object(B.os,'getpgid',return_value=27183),patch.object(B.os,'killpg')as kill:
            with self.assertRaises(OSError):B.command(['unused-cpu-mock'],10,on_started=failed_save)
            kill.assert_called_once_with(27183,signal.SIGTERM);proc.communicate.assert_called_once_with(timeout=5)
    def test_sigterm_handler_cleanup_and_child_mask_restore(self):
        proc=Mock(pid=27184,returncode=None);proc.communicate.return_value=('',None);old={signal.SIGUSR1};calls=[]
        def mask(how,signals):calls.append((how,set(signals)));return old
        def popen(*args,**kwargs):kwargs['preexec_fn']();return proc
        def term(process):signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        previous={s:signal.getsignal(s)for s in(signal.SIGINT,signal.SIGTERM)}
        with patch.object(B.subprocess,'Popen',side_effect=popen),patch.object(B.os,'getpgid',return_value=27184),patch.object(B.os,'killpg')as kill,patch.object(B.signal,'pthread_sigmask',side_effect=mask):
            with self.assertRaises(KeyboardInterrupt):B.command(['unused-cpu-mock'],10,on_started=term)
            kill.assert_called_once_with(27184,signal.SIGTERM)
        self.assertEqual(calls[0],(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM}));self.assertEqual(calls[1],(signal.SIG_SETMASK,old));self.assertEqual(previous,{s:signal.getsignal(s)for s in previous})

if __name__=='__main__':unittest.main()
