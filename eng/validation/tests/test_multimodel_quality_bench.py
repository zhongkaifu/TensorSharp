import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("matrix", Path(__file__).resolve().parents[1] / "multimodel-quality-bench.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class QualityBenchTests(unittest.TestCase):
    def test_linux_sampler_observes_server_membership_and_keeps_other_metrics_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            sampler = matrix.Sampler(Path(directory)/'memory.jsonl')
            counters = SimpleNamespace(_asdict=lambda: {'rss': 123})
            sampler.process = SimpleNamespace(pid=71, memory_info=lambda: counters,
                io_counters=lambda: counters, cpu_times=lambda: counters)
            gpu = SimpleNamespace(returncode=0, stdout='0,10,100,0\n')
            with patch.object(matrix.sys, 'platform', 'linux'), \
                 patch.object(matrix.subprocess, 'run', return_value=gpu), \
                 patch.object(matrix.CGROUP, 'capture', return_value={'memory.current':'50'}) as capture:
                sampler.sample()
                capture.assert_called_once_with(71)
            with patch.object(matrix.sys, 'platform', 'linux'), \
                 patch.object(matrix.subprocess, 'run', return_value=gpu), \
                 patch.object(matrix.CGROUP, 'capture', side_effect=FileNotFoundError('process exited')):
                sampler.sample()
            self.assertEqual(sampler.rows[0]['cgroup']['memory.current'], '50')
            self.assertEqual(sampler.rows[1]['memory']['rss'], 123)
            self.assertEqual(sampler.rows[1]['gpu'][0]['used_mib'], 10)
            self.assertTrue(sampler.rows[1]['cgroup']['errors'])

    def test_openmp_settings_are_explicit_and_inherited_tuning_is_removed(self):
        base = dict(OMP_NUM_THREADS="16", TS_GGML_CPU_THREADS="16")
        inherited = dict(PATH="runtime", OMP_NUM_THREADS="96", OMP_PROC_BIND="true",
                         GOMP_SPINCOUNT="1000000", TS_OTHER="1", TENSORSHARP_OTHER="1")
        env, settings = matrix.backend_environment(base, ["OMP_WAIT_POLICY=PASSIVE", "GOMP_SPINCOUNT=0"], inherited)
        self.assertEqual(settings, {**base, "OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0"})
        self.assertEqual(env, {**settings, "PATH": "runtime"})
        self.assertEqual(base, dict(OMP_NUM_THREADS="16", TS_GGML_CPU_THREADS="16"))
        clean, _ = matrix.backend_environment(base, [], inherited)
        self.assertNotIn("GOMP_SPINCOUNT", clean)

    def test_environment_cannot_override_core_settings_or_hide_duplicates(self):
        for overrides in (["OMP_NUM_THREADS=8"], ["GOMP_SPINCOUNT=0", "GOMP_SPINCOUNT=1"], ["PATH=wrong"], ["OMP_WAIT_POLICY"]):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                matrix.backend_environment(dict(OMP_NUM_THREADS="16"), overrides, {})

    def test_split_identity_requires_every_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / 'model-00001-of-00002.gguf'
            second = root / 'model-00002-of-00002.gguf'
            first.write_bytes(b'one')
            with self.assertRaisesRegex(ValueError, 'Missing'):
                matrix.checkpoint_files(first)
            second.write_bytes(b'two')
            self.assertEqual(matrix.checkpoint_files(first), [first.resolve(), second.resolve()])
            with self.assertRaisesRegex(ValueError, 'shard 1'):
                matrix.checkpoint_files(second)

    def test_prior_verification_cannot_hide_wrong_size_or_incomplete_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, manifest = root / 'model.gguf', root / 'verified.json'
            model.write_bytes(b'model')
            digest = matrix.sha(model)
            data = dict(status='verified', shards=[dict(path=str(model.resolve()), bytes=5, verified=True,
                                                      actual_sha256=digest, sha256=digest)])
            manifest.write_text(json.dumps(data))
            result = matrix.verified_identities([model], manifest)
            self.assertEqual(result[0]['hash_source'], 'prior_whole_file_verification')
            model.write_bytes(b'wrong size')
            with self.assertRaises(ValueError):
                matrix.verified_identities([model], manifest)
            data['status'] = 'running'
            manifest.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                matrix.verified_identities([model], manifest)

    def test_exact_oracles_reject_repairs_and_truncation(self):
        valid = {
            "squares": ",".join(str(i*i) for i in range(1, 21)),
            "tool_json": '{"name":"get_weather","arguments":{"city":"Hangzhou","unit":"celsius"}}',
            "code": "def square(x):\n    return x*x",
            "long_extract": "262", "image_ocr": "4821",
        }
        for case, answer in valid.items():
            with self.subTest(case=case):
                self.assertTrue(matrix.quality(case, answer, True))
                self.assertFalse(matrix.quality(case, answer, False))
                self.assertFalse(matrix.quality(case, '```\n' + answer + '\n```', True))
                self.assertFalse(matrix.quality(case, answer + '\nExplanation.', True))

    def test_bad_json_and_unsafe_or_wrong_code(self):
        self.assertFalse(matrix.quality("tool_json", '{"name":"wrong","name":"get_weather","arguments":{"city":"Hangzhou","unit":"celsius"}}', True))
        for source in ("def square(x):\n    return x+x", "import os\ndef square(x):\n    return x*x", "def square(x=exec('print(1)')):\n    return x*x"):
            self.assertFalse(matrix.quality("code", source, True))

    def test_timings_require_real_uncached_work(self):
        response = dict(prompt_eval_count=100, eval_count=10, prompt_eval_duration=100_000_000,
                        eval_duration=200_000_000, total_duration=400_000_000, prompt_cache_hit_tokens=0)
        self.assertEqual(matrix.metrics(response)["prefill_tps"], 1000)
        self.assertEqual(matrix.metrics(response)["decode_tps"], 50)
        for key, value in (("prompt_cache_hit_tokens", 1), ("total_duration", 1), ("eval_duration", 0), ("eval_count", True), ("prompt_eval_count", -1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                matrix.metrics({**response, key: value})

    def test_first_and_warm_do_not_pool_or_hide_failure(self):
        records = [dict(case="squares", repetition=i, quality_passed=i != 1, metrics=dict(prefill_tps=p, decode_tps=p))
                   for i, p in enumerate((1, 20, 30))]
        result = matrix.summarize(records)["squares"]
        self.assertEqual(result["quality_passes"], 2)
        self.assertEqual(result["first"]["prefill_tps"], 1)
        self.assertEqual(result["warm"]["prefill_tps"]["median"], 25)


if __name__ == "__main__":
    unittest.main()
