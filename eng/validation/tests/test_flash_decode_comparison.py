import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('flash_decode', Path(__file__).parents[1] / 'compare-flash-decode-runs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FlashDecodeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = []
        row = dict(prefill_tokens=4, decode_tokens=2, prefill_ms=4, decode_ms=20,
                   prefill_tps=1000, decode_tps=100, logit_rows=3,
                   full_logit_chain_sha256='all-rows', final_logit_sha256='last-row',
                   generated_tokens=[], decode_step_ms=[10,10],
                   cache_stats_after_prefill={'Calls':0},cache_stats_after_decode={'Calls':4})
        for i in range(4):
            p = Path(self.tmp.name) / str(i)
            p.mkdir()
            execution = dict(exit_code=0, timed_out=False, started_unix=100+i*20, wall_seconds=10,
                             windows_process_memory={'peak_rss_bytes':123}, gpu_sampling={'peaks':[{'peak_memory_used_mib':456}]})
            model = dict(passed=True,run_complete=True,native_sha256='control' if i in (0,3) else 'candidate',
                         checkpoint_identity={'files':['same']}, model_geometry={'context':32,'layers':2},
                         prompt_tokens=[1,2,3,4],forced_tokens=[5,6],decode_mode='teacher-forced',
                         managed_assemblies_sha256={'TensorSharp.Models.dll':'same'},environment={'cache':'same'},
                         requested_options={'output':str(p),'iterations':'2'},
                         cache_stats={'ReservedBytes':100,'BudgetBytes':200,'Hits':6,'Misses':18},
                         cleanup={**dict.fromkeys(('model_disposed','cache_cleared','reuse_released','native_shutdown'),True),
                                  'scope_detached':False,'retained_model_owner':False,'retained_scope_owner':False,'errors':[]},
                         runs=[dict(row,warmup=warm) for warm in (True,False,False)])
            (p/'execution.json').write_text(json.dumps(execution))
            (p/'model.json').write_text(json.dumps(model))
            (p/'process.log').write_text('' if i in (0,3) else
                '[HOSTMOE-CACHE-FILE] layer=0 calls=2 file_bytes=512\n'
                '[HOSTMOE-FILE-WORKSPACE] bytes=1024 ceiling=33554432 type=CUDA_Host\n')
            self.paths.append(p/'execution.json')

    def change(self, which, callback, file='model.json'):
        p = self.paths[which].parent / file
        d=json.loads(p.read_text());callback(d);p.write_text(json.dumps(d))

    def test_balanced_complete(self):
        result=module.compare(self.paths)
        self.assertTrue(result['passed'],result['failures'])
        self.assertEqual(result['measurements']['candidate']['decode_tps']['samples'],4)

    def test_explicit_file_read_comparison(self):
        for i in range(4):
            self.change(i, lambda d: d['environment'].update(TS_HOST_MOE_FILE_READ='0' if i in (0,3) else '1'))
        self.assertTrue(module.compare(self.paths, candidate_file_read=True)['passed'])
        self.assertFalse(module.compare(self.paths)['passed'])
        self.change(1, lambda d: d['environment'].update(TS_HOST_MOE_FILE_READ='0'))
        self.assertFalse(module.compare(self.paths, candidate_file_read=True)['passed'])

    def test_file_read_comparison_keeps_prefetch_equal(self):
        for i in range(4):
            self.change(i, lambda d: d['environment'].update(TS_HOST_MOE_FILE_READ='0' if i in (0,3) else '1'))
        self.change(1, lambda d: d['environment'].update(TS_HOST_MOE_EXPERT_CACHE_PREFETCH='0'))
        self.assertFalse(module.compare(self.paths, candidate_file_read=True)['passed'])

    def test_file_read_requires_actual_engagement_and_bounded_workspace(self):
        for i in range(4):
            self.change(i, lambda d: d['environment'].update(TS_HOST_MOE_FILE_READ='0' if i in (0,3) else '1'))
        path = self.paths[1].parent/'process.log'
        path.write_text('')
        self.assertFalse(module.compare(self.paths, candidate_file_read=True)['passed'])
        path.write_text('[HOSTMOE-CACHE-FILE] layer=0 calls=2 file_bytes=512\n'
            '[HOSTMOE-FILE-WORKSPACE] bytes=33554433 ceiling=33554432\n')
        self.assertFalse(module.compare(self.paths, candidate_file_read=True)['passed'])

    def test_windows_binary_paths_compare_portably(self):
        for i in range(4):
            self.change(i,lambda d:d.update(managed_assemblies_sha256={f'C:\\arm{i}\\TensorSharp.Models.dll':'same'}))
        self.assertTrue(module.compare(self.paths)['passed'])

    def prepare_lfu(self):
        for i in range(4):
            candidate = i in (1,2)
            self.change(i, lambda d: d['environment'].update(TS_HOST_MOE_FILE_READ='1', TS_HOST_MOE_EXPERT_CACHE_LFU='1' if candidate else '0'))
            (self.paths[i].parent/'process.log').write_text(
                '[HOSTMOE-CACHE-FILE] layer=0 calls=2 file_bytes=512\n'
                '[HOSTMOE-FILE-WORKSPACE] bytes=1024 ceiling=33554432\n'
                + f'[HOSTMOE-EVICTION] layer=0 policy={"lfu" if candidate else "lru"} epoch={256 if candidate else 0}\n')

    def test_lfu_requires_both_file_paths_and_actual_policy(self):
        self.prepare_lfu()
        self.assertTrue(module.compare(self.paths, candidate_lfu=True)['passed'])
        path = self.paths[1].parent/'process.log'
        path.write_text(path.read_text().replace('policy=lfu', 'policy=lru'))
        self.assertFalse(module.compare(self.paths, candidate_lfu=True)['passed'])

    def test_lfu_does_not_allow_changing_file_reads_or_prefetch(self):
        self.prepare_lfu()
        self.change(1, lambda d: d['environment'].update(TS_HOST_MOE_FILE_READ='0'))
        self.assertFalse(module.compare(self.paths, candidate_lfu=True)['passed'])
        self.change(1, lambda d: d['environment'].update(TS_HOST_MOE_FILE_READ='1', TS_HOST_MOE_EXPERT_CACHE_PREFETCH='0'))
        self.assertFalse(module.compare(self.paths, candidate_lfu=True)['passed'])

    def test_lfu_cannot_be_combined_with_transport_experiment(self):
        self.prepare_lfu()
        self.assertFalse(module.compare(self.paths, candidate_file_read=True, candidate_lfu=True)['passed'])

    def test_rejects_final_row_only(self):
        self.change(1,lambda d:d['runs'][1].pop('full_logit_chain_sha256'))
        self.assertFalse(module.compare(self.paths)['passed'])

    def test_rejects_changed_warmup(self):
        self.change(1,lambda d:d['runs'][0].update(full_logit_chain_sha256='changed'))
        self.assertFalse(module.compare(self.paths)['passed'])

    def test_rejects_wrong_timing_denominator(self):
        self.change(2,lambda d:d['runs'][1].update(decode_tokens=3))
        self.assertFalse(module.compare(self.paths)['passed'])

    def test_rejects_overlap(self):
        self.change(1,lambda d:d.update(started_unix=105),'execution.json')
        self.assertFalse(module.compare(self.paths)['passed'])

    def test_rejects_wrong_order_label(self):
        self.change(0,lambda d:d.update(arm='candidate'),'execution.json')
        self.assertFalse(module.compare(self.paths)['passed'])

    def test_rejects_extra_cache(self):
        self.change(1,lambda d:d['cache_stats'].update(ReservedBytes=200))
        self.assertFalse(module.compare(self.paths)['passed'])

    def test_rejects_cpu_fallback(self):
        self.change(1,lambda d:d['runs'][1].update(cache_stats_after_decode={'Calls':3}))
        self.assertFalse(module.compare(self.paths)['passed'])

    def test_rejects_inactive_cache(self):
        for i in range(4):
            self.change(i,lambda d:d.update(cache_stats={'ReservedBytes':0}))
        self.assertFalse(module.compare(self.paths)['passed'])

    def test_rejects_truncated_greedy_answer(self):
        for i in range(4):
            self.change(i,lambda d:(d.update(decode_mode='greedy'),[r.update(finish_reason='length') for r in d['runs']]))
        self.assertFalse(module.compare(self.paths)['passed'])

    def test_rejects_incomplete_cleanup(self):
        self.change(3,lambda d:d['cleanup'].update(cache_cleared=False))
        self.assertFalse(module.compare(self.paths)['passed'])

    def test_accepts_detached_created_scope(self):
        for i in range(4):
            self.change(i, lambda d: (d.update(device_budget_bytes=1024), d['cleanup'].update(scope_detached=True)))
        self.assertTrue(module.compare(self.paths)['passed'])

    def test_rejects_attached_created_scope(self):
        for i in range(4):
            self.change(i, lambda d: d.update(device_budget_bytes=1024))
        self.assertFalse(module.compare(self.paths)['passed'])

    def test_rejects_retained_owner_without_scope(self):
        self.change(1, lambda d: d['cleanup'].update(retained_model_owner=True))
        self.assertFalse(module.compare(self.paths)['passed'])

    def test_rejects_missing_owner_evidence(self):
        self.change(1, lambda d: d['cleanup'].pop('retained_scope_owner'))
        self.assertFalse(module.compare(self.paths)['passed'])

    def test_preserves_regression(self):
        for i in (1,2):
            def slow(d):
                for r in d['runs']:r.update(decode_ms=40,decode_tps=50,decode_step_ms=[20,20])
            self.change(i,slow)
        result=module.compare(self.paths)
        self.assertTrue(result['passed'],result['failures'])
        self.assertEqual(result['candidate_to_control_ratio']['decode_tps'],0.5)

    def test_reverse_order_labels_outer_pair_as_candidate(self):
        for i in (1,2):
            def slow(d):
                for r in d['runs']:r.update(decode_ms=40,decode_tps=50,decode_step_ms=[20,20])
            self.change(i,slow)
        result=module.compare(self.paths,candidate_first=True)
        self.assertTrue(result['passed'],result['failures'])
        self.assertEqual(result['execution_order'],['candidate','control','control','candidate'])
        self.assertEqual(result['candidate_to_control_ratio']['decode_tps'],2)


if __name__=='__main__':unittest.main()
