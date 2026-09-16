"""Local unit fixtures only: no models, builds, SSH, CUDA, or external packages."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import native_release_evidence as evidence


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plans = load('make-native-release-plans')
runner = load('run-native-comparison')
drafts = load('validate-speculative-rows')


def draft_rows():
    return [{'Scenario': 'spec', 'Label': label, 'Tokens': [11, 22, 33], 'Finish': 'eos',
             'Drafted': 5, 'Accepted': 2, 'VerifySteps': 2, 'PlainSteps': 1, 'Rollbacks': 1}
            for label in ('plain greedy', 'draft head (auto)')]


class IdentityTests(unittest.TestCase):
    def test_cli_hash_mismatch_stops_before_any_build_or_model_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'models/fixture'
            folder.mkdir(parents=True)
            weight = folder / 'model.gguf'
            weight.write_bytes(b'weight fixture')
            native = root / 'libGgmlOps.so'
            native.write_bytes(b'native fixture')
            catalog = root / 'catalog.json'
            catalog.write_text(json.dumps({'models': [{'id': 'fixture', 'repo': 'publisher/repo', 'modalities': ['text']}]}))
            inventory = root / 'inventory.json'
            inventory.write_text(json.dumps({'models': [{'id': 'fixture', 'repo': 'publisher/repo',
                'status': 'downloaded-and-verified', 'modalities': ['text'], 'files': [{'path': str(weight),
                    'bytes': weight.stat().st_size, 'size': weight.stat().st_size, 'repository': 'publisher/repo',
                    'revision': 'publisher-revision', 'publisher_digest_kind': 'lfs-sha256',
                    'download_verified_sha256': hashlib.sha256(weight.read_bytes()).hexdigest()}]}]}))
            args = ['run-native-comparison.py', '--repo', str(root), '--dotnet', str(root / 'dotnet'),
                    '--baseline', str(native), '--candidate', str(native), '--baseline-sha256', '0' * 64,
                    '--candidate-sha256', '0' * 64, '--inventory', str(inventory),
                    '--inventory-sha256', hashlib.sha256(inventory.read_bytes()).hexdigest(),
                    '--catalog', str(catalog), '--models-root', str(root / 'models'), '--output', str(root / 'result')]
            with mock.patch.object(sys, 'argv', args), mock.patch.object(runner, 'checked_process') as process:
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    runner.main()
                process.assert_not_called()

    def test_shard_count_alone_does_not_establish_complete_checkpoint(self):
        for names in (['a-00001-of-00002.gguf', 'a-00003-of-00002.gguf'],
                      ['a-00001-of-00002.gguf', 'b-00002-of-00002.gguf'],
                      ['a-00001-of-00002.gguf', 'a-00002-of-00003.gguf'],
                      ['a-00001-of-00002.gguf', 'a.gguf'], ['a.gguf', 'b.gguf']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                evidence.validate_shards([{'path': name} for name in names])
        evidence.validate_shards([{'path': f'a-{i:05}-of-00002.gguf'} for i in (2, 1)])

    def test_hash_pin_rejects_current_binary_and_wrong_cached_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'libGgmlOps.so'
            path.write_bytes(b'original native fixture')
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            stat = path.stat()
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                runner.verify_file(path, '0' * 64)
            cached = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': '0' * 64}
            result = runner.verify_file(path, actual, cached)
            self.assertEqual('fresh-whole-file-sha256', result['verification'])
            cached['sha256'] = actual
            self.assertEqual('prior-hash-with-matching-size-and-mtime', runner.verify_file(path, actual, cached)['verification'])

    def test_same_filename_from_different_asset_set_does_not_bind(self):
        model = {'id': 'fixture', 'publisher': 'publisher/repo', 'files': ['/other/model.gguf']}
        inventory = {'fixture': {'repo': 'publisher/repo', 'files': [{'path': '/verified/model.gguf'}]}}
        with self.assertRaisesRegex(ValueError, 'differ from pinned inventory'):
            runner.bind_model_identity(model, inventory)

    def test_actual_mapped_native_digest_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / 'libGgmlOps.so'
            native.write_bytes(b'mapped native fixture')
            digest = hashlib.sha256(native.read_bytes()).hexdigest()
            for reported, expected_ok in ((digest, True), ('0' * 64, False)):
                child = mock.Mock(pid=987654321)
                child.poll.side_effect = [None, 0, 0]
                child.wait.return_value = 0
                maps = f'1000-2000 r-xp 0000 00:00 1 {native}\n'
                with mock.patch.object(runner.subprocess, 'Popen', return_value=child), \
                        mock.patch.object(Path, 'read_text', return_value=maps), \
                        mock.patch.object(runner.time, 'sleep'):
                    result = runner.checked_process(['fixture'], {}, Path(directory) / 'run.log', directory, 1, reported)
                self.assertEqual(expected_ok, result['native_verified'])
                self.assertEqual({str(native): digest}, result['mapped_native_libraries'])

    def test_missing_proc_map_is_unqualified_even_if_process_exits_successfully(self):
        with tempfile.TemporaryDirectory() as directory:
            child = mock.Mock(pid=987654321)
            child.poll.side_effect = [None, 0, 0]
            child.wait.return_value = 0
            with mock.patch.object(runner.subprocess, 'Popen', return_value=child), \
                    mock.patch.object(Path, 'read_text', side_effect=FileNotFoundError), \
                    mock.patch.object(runner.time, 'sleep'):
                result = runner.checked_process(['fixture'], {}, Path(directory) / 'run.log', directory, 1, '0' * 64)
            self.assertEqual(0, result['exit_code'])
            self.assertFalse(result['native_verified'])


class PlacementTests(unittest.TestCase):
    # Literal banners from ModelBase and Gemma4Model/Model.TensorParallel.
    GEMMA_TP = ('Backend: GgmlCuda\n  Gemma4 TP weight sharding complete (2 GPUs).\n'
                '  KV cache: 148 MB (dtype: f16, global layers: 8192 seq, SWA layers: 512 seq)\n')

    def test_requested_flags_and_visible_devices_never_prove_features(self):
        log = '[agent-turn-bench] loading model backend=GgmlCuda kv=q4_0\nggml_cuda_init: found 2 CUDA devices\n'
        result = evidence.placement_evidence(log, 'gemma4-e4b', runner.PROFILES['gpu2-f16'])
        self.assertEqual('unqualified', result['status'])
        self.assertEqual([], result['effective_tp_degrees'])
        self.assertEqual([], result['effective_kv_types'])

    def test_completed_sharding_and_effective_cache_qualify_tp(self):
        self.assertEqual('passed', evidence.placement_evidence(self.GEMMA_TP, 'gemma4-e4b', runner.PROFILES['gpu2-f16'])['status'])

    def test_single_kernel_decline_does_not_remove_real_per_op_tp(self):
        log = self.GEMMA_TP + 'fused TP decode unavailable; using per-op TP path\n'
        self.assertEqual('passed', evidence.placement_evidence(log, 'gemma4-e4b', runner.PROFILES['gpu2-f16'])['status'])

    def test_wrong_cache_or_tp_degree_stays_unqualified(self):
        for log in (self.GEMMA_TP.replace('dtype: f16', 'dtype: q4_0'), self.GEMMA_TP.replace('(2 GPUs)', '(1 GPUs)')):
            result = evidence.placement_evidence(log, 'gemma4-e4b', runner.PROFILES['gpu2-f16'])
            self.assertEqual('unqualified', result['status'])
            self.assertTrue(result['contradictions'])

    def test_layer_split_is_not_tensor_parallel_and_empty_device_does_not_count(self):
        log = 'Backend: GgmlCuda\nLayer split across 2 GPUs: gpu0=20 layers/100 MB, gpu1=20 layers/100 MB\n'
        tp = evidence.placement_evidence(log, 'qwen35-9b', runner.PROFILES['gpu2-f16'])
        layer = evidence.placement_evidence(log, 'qwen38', runner.PROFILES['gpu2-f16'])
        self.assertTrue(tp['contradictions'])
        self.assertEqual('layer-split', layer['requested_mode'])
        self.assertEqual({0: 20, 1: 20}, layer['layer_counts_by_device'])
        empty = evidence.placement_evidence(log.replace('gpu1=20', 'gpu1=0'), 'qwen38', runner.PROFILES['gpu2-f16'])
        self.assertTrue(any('every assigned GPU' in item for item in empty['missing_evidence']))

    def test_partial_device_preload_is_not_all_layer_residency(self):
        log = ('Backend: GgmlCuda\nggml_cuda_init: found 1 CUDA devices\n'
               'Device-resident quantized weights: 1819 MB across 50 tensors\n'
               '[agent-turn-bench] effective managed KV storage dtype=f16 model_type=GptOssModel\n')
        result = evidence.placement_evidence(log, 'gptoss20b', runner.PROFILES['gpu1-f16'])
        self.assertEqual('passed', result['status'])
        self.assertTrue(result['permanent_weight_residency'].startswith('unknown'))

    def test_generic_property_cannot_qualify_opaque_native_cache(self):
        log = ('Backend: GgmlCuda\nggml_cuda_init: found 1 CUDA devices\n'
               '[agent-turn-bench] effective managed KV storage dtype=f16 model_type=GlmDsaModel\n')
        result = evidence.placement_evidence(log, 'glm52', runner.PROFILES['gpu1-f16'])
        self.assertEqual([], result['effective_kv_types'])
        self.assertEqual('unqualified', result['status'])

    def test_offload_configuration_does_not_prove_expert_placement(self):
        result = evidence.placement_evidence('Backend: GgmlCuda\nTS_N_CPU_MOE=4\n', 'gptoss20b', runner.PROFILES['cpu-moe4'])
        self.assertEqual([], result['effective_offload_layer_counts'])
        log = "[glm] MoE CPU offload: routed experts of layers 0..3 (1.0 GiB) stay in system RAM\n"
        result = evidence.placement_evidence(log, 'glm52', runner.PROFILES['cpu-moe4'])
        self.assertEqual([4], result['effective_offload_layer_counts'])
        self.assertEqual('unqualified', result['status'])  # Other required evidence is absent.


class DrafterTests(unittest.TestCase):
    def test_engagement_and_exact_parity_are_both_required(self):
        self.assertEqual('passed', drafts.validate(draft_rows())['status'])
        for edit in ({'Drafted': 0}, {'Accepted': 0}, {'VerifySteps': 0}, {'Accepted': 6},
                     {'Drafted': float('nan')}, {'Accepted': True}, {'Rollbacks': -1},
                     {'Tokens': [11, 24, 33]}, {'Tokens': []}, {'Finish': 'length'}):
            rows = draft_rows()
            rows[1].update(edit)
            with self.subTest(edit=edit):
                self.assertEqual('failed', drafts.validate(rows)['status'])

    def test_ngram_and_attachment_fallback_do_not_count_as_learned_drafting(self):
        rows = draft_rows()
        rows[1]['Label'] = 'ngram'
        self.assertEqual('failed', drafts.validate(rows)['status'])
        with self.assertRaises(ValueError):
            drafts.validate(draft_rows() + [draft_rows()[1]])

    def test_later_inactive_pass_is_not_hidden_by_first_row_median(self):
        samples = [draft_rows(), draft_rows()]
        samples[1][1]['VerifySteps'] = 0
        comparator = mock.Mock()
        comparator.load_rows.side_effect = [{(row['Scenario'], row['Label']): row for row in rows} for rows in samples]
        result = runner.validate_learned_samples(Path('candidate.json'), 2, comparator, drafts)
        self.assertEqual('failed', result['status'])
        self.assertEqual(2, len(result['samples']))
        self.assertTrue(any('measure2.json' in reason for reason in result['failures']))


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory_path = SCRIPTS.parents[1] / 'docs/validation/ggml-no-patch-2026-09-15/download-inventory-complete.json'
        cls.inventory = json.loads(cls.inventory_path.read_text())

    def make(self, inventory=None):
        return plans.build_plan(inventory or self.inventory, 'a' * 64, '/workspace/release',
                                self.inventory_path, 'b' * 64)

    def test_every_runnable_comparison_pins_native_and_publisher_inventory(self):
        plan = self.make()
        self.assertFalse(plan['executed'])
        for cell in plan['comparisons'] + plan['drafters'] + plan['capacity_review']:
            self.assertFalse(cell['executed'])
            self.assertFalse(cell['qualified'])
            command = cell.get('command', cell.get('command_after_capacity_review'))
            if command:
                for flag, expected in (('--candidate-sha256', 'a' * 64), ('--baseline-sha256', plans.BASELINE_SHA256),
                                       ('--inventory-sha256', 'b' * 64)):
                    self.assertEqual(expected, command[command.index(flag) + 1])
                self.assertNotIn('\\', command[1])

    def test_drafter_catalog_retains_all_eight_companions_and_unavailable_flash_mtp(self):
        rows = {row['id']: row for row in self.make()['drafters']}
        self.assertEqual(13, len(rows))
        self.assertEqual(set(plans.DRAFTS), {key for key in rows if key in plans.DRAFTS})
        self.assertEqual('unavailable', rows['glm53-flash-embedded-mtp']['status'])
        self.assertNotIn('command', rows['glm53-flash-embedded-mtp'])
        self.assertEqual('capacity-review-required', rows['deepseek4-dspark']['status'])
        self.assertNotIn('command', rows['deepseek4-dspark'])
        for target in ('glm52', 'glm53'):
            self.assertIn('--enable-native-speculation', rows[target + '-embedded-mtp']['command_after_capacity_review'])

    def test_dense_offload_and_unsupported_translation_tp_are_not_prepared(self):
        cells = {(row['model'], row['profile']) for row in self.make()['comparisons']}
        self.assertNotIn(('muse-glimmer', 'cpu-moe4'), cells)
        self.assertNotIn(('hunyuan-dense', 'gpu2-f16'), cells)
        self.assertIn(('qwen38', 'gpu2-f16'), cells)
        self.assertNotIn('muse-glimmer', runner.MOE_MODELS)

    def test_ambiguous_inventory_or_missing_publisher_digest_rejected(self):
        for mutation in ('duplicate', 'digest', 'size'):
            inventory = copy.deepcopy(self.inventory)
            if mutation == 'duplicate':
                inventory['models'].append(inventory['models'][0])
            elif mutation == 'digest':
                del inventory['models'][0]['files'][0]['download_verified_sha256']
            else:
                inventory['models'][0]['files'][0]['size'] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.make(inventory)


if __name__ == '__main__':
    unittest.main()
