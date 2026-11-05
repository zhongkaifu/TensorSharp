import contextlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

TOOL = Path(__file__).resolve().parents[1] / 'qwen38-optional-engine-setup.py'
spec = importlib.util.spec_from_file_location('optional_setup', TOOL)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class OptionalSetupTests(unittest.TestCase):
    def test_default_plan_never_creates_output_or_runs_commands(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / 'fresh'
            with patch.object(m.subprocess, 'Popen', side_effect=AssertionError('execution')), contextlib.redirect_stdout(io.StringIO()) as text:
                self.assertEqual(m.main(['--engine', 'eole', '--out', str(out)]), 0)
            self.assertFalse(out.exists())
            plan = json.loads(text.getvalue())
            self.assertEqual(plan['execution'], 'NOT_RUN'); self.assertEqual(plan['quality'], 'NOT_RUN')
            self.assertTrue(plan['setup_only'])

    def test_resolver_rejects_all_protected_dependency_changes(self):
        for name in ('torch', 'triton', 'nvidia-cublas-cu13', 'NVIDIA_CUDA_RUNTIME_CU13'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'protected'):
                m.resolved_requirements({'install': [dict(metadata=dict(name=name, version='new'), download_info={})]})

    def test_resolver_requires_https_exact_hash_and_skips_pinned_local_eole(self):
        def row(url, hashes):
            return dict(metadata=dict(name='helper', version='1.2'), download_info=dict(url=url, archive_info=dict(hashes=hashes)))
        for item in (row('file:///tmp/x.whl', {'sha256': 'a'*64}), row('https://example.org/a.whl', {})):
            with self.assertRaises(ValueError): m.resolved_requirements({'install': [item]})
        actual = m.resolved_requirements({'install': [dict(metadata=dict(name='eole')), row('https://example.org/a.whl', {'sha256': 'a'*64})]})
        self.assertEqual(actual, [dict(name='helper', version='1.2', url='https://example.org/a.whl', sha256='a'*64)])

    def test_profiles_keep_plain_spec_precision_context_and_independent_outs(self):
        for engine in ('eole', 'focus-llama'):
            args = m.parser().parse_args(['--engine', engine, '--out', '/root/fresh'])
            rows = m.profiles(args, {'MODEL_MANIFEST_JSON': '/root/fresh/model-manifest.json'})
            self.assertEqual(len(rows), 4); self.assertEqual(len({x['environment']['OUT'] for x in rows}), 4)
            self.assertEqual([x['expected_cases'] for x in rows], [48, 6, 48, 6])
            self.assertTrue(all(x['quality'] == 'NOT_RUN' for x in rows))
            self.assertEqual([x['environment']['CTX'] for x in rows], ['40960','262144','40960','262144'])
            self.assertEqual([x['environment']['LANES'] for x in rows], ['1','1','1','1'] if engine=='eole' else ['4','1','4','1'])
            self.assertTrue(all('--allow-failures' in x['bench_argv'] for x in rows))
            self.assertTrue(all(x['label'].startswith('vm2-') for x in rows))
            self.assertTrue(all(x['environment']['HAYSTACK']=='/workspace/ts-eval/artifacts/haystack.txt' for x in rows))
            self.assertEqual(args.bench_python, Path('/root/venvs/bench/bin/python'))

    def test_output_cannot_overlap_shared_sg_or_harness(self):
        args = m.parser().parse_args(['--engine', 'eole', '--out', '/root/venvs/sglang/bad'])
        with self.assertRaisesRegex(ValueError, 'protected'): m.output_guard(args)
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'real').mkdir();(p/'alias').symlink_to(p/'real', target_is_directory=True)
            args.out=p/'alias'/'new'
            with self.assertRaisesRegex(ValueError, 'symlink'):m.output_guard(args)

    def test_inventory_records_links_without_traversing_external_directory(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'owned').mkdir();(p/'external').mkdir();(p/'external/x').write_text('external')
            (p/'owned/link').symlink_to(p/'external', target_is_directory=True)
            (p/'owned/file').write_text('inside')
            rows=m.inventory(p/'owned')
            self.assertEqual([r['path'] for r in rows], ['file','link'])
            self.assertIn('symlink', rows[1]);self.assertNotIn('sha256', rows[1])

    def test_timeout_cleanup_only_kills_matching_owned_group(self):
        class Child:
            pid=12345
            stopped=False
            def wait(self, timeout):
                if not self.stopped:raise subprocess.TimeoutExpired(['owned'],timeout)
                return -signal.SIGTERM
            def poll(self):return -signal.SIGTERM if self.stopped else None
        for same in (True, False):
            with self.subTest(same=same), tempfile.TemporaryDirectory() as root:
                child=Child();kills=[];ids=iter([{'pid':12345,'starttime':'10'}, {'pid':12345,'starttime':'10' if same else '11'}])
                def kill(group,sig):kills.append((group,sig));child.stopped=True
                run=m.Runner(Path(root),30,1,{})
                with patch.object(m.subprocess,'Popen',return_value=child),patch.object(m,'proc_identity',side_effect=lambda _:next(ids)),patch.object(m.os,'getpgid',return_value=12345),patch.object(m.os,'killpg',side_effect=kill):
                    with self.assertRaises(subprocess.TimeoutExpired):run.call('test',['owned'])
                self.assertEqual(kills,[(12345,signal.SIGTERM)] if same else [])
                self.assertEqual(json.loads((Path(root)/'events.json').read_text())[0]['status'],'FAILED_OR_INTERRUPTED')
                self.assertEqual(json.loads((Path(root)/'events.json').read_text())[0]['cleanup_group_signaled'],same)

    def test_registration_failure_still_cleans_new_live_popen_child(self):
        class Child:
            pid=12345
            stopped=False
            def wait(self,timeout):return -signal.SIGTERM
            def poll(self):return -signal.SIGTERM if self.stopped else None
        with tempfile.TemporaryDirectory() as root:
            child=Child();kills=[]
            def kill(group,sig):kills.append((group,sig));child.stopped=True
            with patch.object(m.subprocess,'Popen',return_value=child),patch.object(m,'proc_identity',side_effect=OSError('registration failed')),patch.object(m.os,'getpgid',return_value=child.pid),patch.object(m.os,'killpg',side_effect=kill):
                with self.assertRaisesRegex(OSError,'registration failed'):m.Runner(Path(root),30,1,{}).call('registration',['owned'])
            self.assertEqual(kills,[(child.pid,signal.SIGTERM)])
            self.assertEqual(json.loads((Path(root)/'events.json').read_text())[0]['status'],'FAILED_OR_INTERRUPTED')


if __name__ == '__main__':unittest.main()
