import importlib.util
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import threading
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('maya_token_baseline', Path(__file__).parents[1] / 'maya-token-baseline.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class MayaWireTests(unittest.TestCase):
    def test_prefill_default_and_explicit_capacity(self):
        self.assertEqual('auto', mod.prefill_argument({}))
        self.assertEqual('auto', mod.prefill_argument({'prefill': 'auto'}))
        for value in (32, '32', '0032'):
            self.assertEqual('32', mod.prefill_argument({'prefill': value}))

    def test_prefill_rejects_ambiguous_or_unbounded_capacity(self):
        for value in (0, -1, True, 32.0, '32junk', '0', 'AUTO', ' 32', '３２', 2147483648, None):
            with self.assertRaises(ValueError, msg=repr(value)):
                mod.prefill_argument({'prefill': value})

    def reference(self):
        return dict(name='math', prompt=mod.fixed_tasks.PROMPTS['math'], rendered_prompt='<|assistant|>',
                    parsed_text='42', reasoning_text='', generated_tokens=[42, 9], prompt_tokens=[1, 2, 3], passed=True, eos=True)

    def row(self):
        return dict(mod.parse_done('DONE 2 3 120.0 20.0 stop 0 0 0 15 20'), tokens=[42, 9])

    def test_complete_exact_eos_requires_independent_semantics(self):
        self.assertFalse(mod.qualify(self.row(), self.reference(), [9])['passed'])
        self.assertTrue(mod.qualify(self.row(), self.reference(), [9], dict(passed=True))['passed'])

    def test_length_never_proves_eos_completion(self):
        row = self.row(); row['finish'] = 'length'
        self.assertFalse(mod.qualify(row, self.reference(), [9])['passed'])

    def test_earlier_eos_is_rejected(self):
        row = self.row(); row['tokens'] = [9, 9]
        ref = self.reference(); ref['generated_tokens'] = [9, 9]
        self.assertFalse(mod.qualify(row, ref, [9])['passed'])

    def test_truncated_or_extra_wire_tokens_fail(self):
        for key, value in [('generated', 3), ('prompt_tokens', 4), ('reused', 1), ('drafts_offered', 1)]:
            row = self.row(); row[key] = value
            self.assertFalse(mod.qualify(row, self.reference(), [9])['passed'], key)

    def test_different_tokens_do_not_infer_semantics(self):
        row = self.row(); row['tokens'] = [43, 9]
        result = mod.qualify(row, self.reference(), [9])
        self.assertFalse(result['passed'])
        self.assertFalse(result['semantic_inferred_from_exact_reference'])
        self.assertTrue(result['checkpoint_eos_observed'])

    def test_reference_semantic_flag_does_not_grade_candidate(self):
        ref = self.reference(); ref['passed'] = False
        self.assertFalse(mod.qualify(self.row(), ref, [9])['passed'])
        self.assertTrue(mod.qualify(self.row(), ref, [9], dict(passed=True))['passed'])
        ref['eos'] = False
        self.assertFalse(mod.qualify(self.row(), ref, [9], dict(passed=True))['passed'])

    def test_separate_answer_check_can_pass_different_tokens(self):
        row = self.row(); row['tokens'] = [43, 9]
        result = mod.qualify(row, self.reference(), [9], dict(passed=True))
        self.assertTrue(result['passed'])
        self.assertFalse(result['exact_reference_tokens'])
        self.assertFalse(result['semantic_inferred_from_exact_reference'])

    def test_parser_failure_cannot_be_overridden_by_exact_tokens(self):
        result = mod.qualify(self.row(), self.reference(), [9], dict(passed=False))
        self.assertFalse(result['passed'])

    def test_parser_pass_cannot_override_missing_eos(self):
        row = self.row(); row['tokens'] = [42, 8]
        self.assertFalse(mod.qualify(row, self.reference(), [9], dict(passed=True))['passed'])

    def test_production_parser_result_uses_independent_fixed_task(self):
        class PlainParser:
            def __init__(self, **kwargs): pass
            def feed(self, raw): return [SimpleNamespace(kind='content', text=raw)]
            def finish(self): return []
        reference = self.reference()
        reference['parsed_text'] = '41'
        self.assertTrue(mod.semantic_result('42', reference, PlainParser)['passed'])
        self.assertFalse(mod.semantic_result('41', reference, PlainParser)['passed'])
        reference.update(name='extraction', prompt=mod.fixed_tasks.PROMPTS['extraction'], parsed_text='cedar, maple')
        for answer, expected in [('cedar,maple', True), ('cedar, maple', True),
                                 ('["cedar","maple"]', False), ('```\ncedar,maple\n```', False)]:
            result = mod.semantic_result(answer, reference, PlainParser)
            self.assertEqual(expected, result['passed'])
            self.assertEqual(mod.fixed_tasks.POLICY, result['policy'])
        reference.pop('prompt')
        self.assertFalse(mod.semantic_result('cedar, maple', reference, PlainParser)['passed'])

    def test_unclosed_thinking_cannot_pass_correct_fixed_answer(self):
        class PlainParser:
            def __init__(self, **kwargs): pass
            def feed(self, raw): return [SimpleNamespace(kind='content', text=raw)]
            def finish(self): return []
        reference = self.reference(); reference['rendered_prompt'] += '<think>'
        self.assertFalse(mod.semantic_result('42', reference, PlainParser)['passed'])

    def test_extended_fixed_tasks_keep_semantics_eos_and_structure_separate(self):
        class PlainParser:
            def __init__(self, **kwargs): pass
            def feed(self, raw): return [SimpleNamespace(kind='content', text=raw)]
            def finish(self): return []
        for name, answer in [('code', 'def square(x):\n    return x*x'),
                             ('tool_json', '{"name":"get_weather","arguments":{"city":"Hangzhou","unit":"celsius"}}')]:
            reference = self.reference()
            reference.update(name=name, prompt=mod.fixed_tasks.PROMPTS[name], parsed_text='wrong reference answer', passed=False)
            semantic = mod.semantic_result(answer, reference, PlainParser)
            self.assertTrue(semantic['passed'])
            self.assertTrue(mod.qualify(self.row(), reference, [9], semantic)['passed'])
            row = self.row(); row['finish'] = 'length'
            self.assertFalse(mod.qualify(row, reference, [9], semantic)['passed'])
            row = self.row(); row['tokens'] = [42, 8]
            self.assertFalse(mod.qualify(row, reference, [9], semantic)['passed'])
            wrong = mod.semantic_result('```\n' + answer + '\n```', reference, PlainParser)
            self.assertFalse(wrong['passed'])
            self.assertFalse(mod.qualify(self.row(), reference, [9], wrong)['passed'])

    def test_nonfinite_negative_unknown_done_rejected(self):
        for line in ['DONE 2 3 nan 20 stop 0 0 0 15 20', 'DONE 2 3 20 inf stop 0 0 0 15 20',
                     'DONE -1 3 20 20 stop 0 0 0 15 20', 'DONE 2 3 20 20 unknown 0 0 0 15 20', 'DONE 2 3 20 20 stop']:
            with self.assertRaises(ValueError, msg=line): mod.parse_done(line)

    def test_cleanup_unstarted_threads_does_not_skip_process(self):
        process = Mock(); process.poll.side_effect = [None, 1]
        timer = threading.Timer(1, lambda: None)
        reader = threading.Thread(target=lambda: None)
        sampler = SimpleNamespace(stop=threading.Event(), thread=threading.Thread(target=lambda: None))
        code, errors = mod.cleanup_owned(process, timer, reader, sampler)
        self.assertEqual(1, code)
        self.assertEqual([], errors)
        process.kill.assert_called_once()
        self.assertTrue(sampler.stop.is_set())

    def test_cleanup_join_and_kill_failures_do_not_skip_sampler(self):
        process = Mock(); process.poll.return_value = None; process.kill.side_effect = OSError('kill error')
        process.wait.side_effect = TimeoutError('wait error')
        reader = Mock(ident=1); reader.join.side_effect = RuntimeError('join error'); reader.is_alive.return_value = True
        sampler = SimpleNamespace(stop=threading.Event(), thread=threading.Thread(target=lambda: None))
        code, errors = mod.cleanup_owned(process, None, reader, sampler)
        self.assertIsNone(code)
        self.assertTrue(any('kill error' in x for x in errors))
        self.assertTrue(any('join error' in x for x in errors))
        self.assertTrue(sampler.stop.is_set())


class MayaRelocationTests(unittest.TestCase):
    """Actual tiny verified copies; never open a model, build, or launch Maya."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='maya-relocation-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.original_root, self.runtime_root, self.receipts = (self.root / name for name in ('original', 'runtime', 'receipts'))
        for path in (self.original_root, self.runtime_root, self.receipts): path.mkdir()
        self.original_model, self.runtime_model = self.original_root / 'glm', self.runtime_root / 'glm'
        self.original_model.mkdir()
        self.original_pack, self.runtime_pack = self.original_model / 'maya-pack', self.runtime_model / 'maya-pack'
        self.original_pack.mkdir()
        model_rows, pack_rows = [], []
        for name in ('part-00001-of-00003.gguf', 'part-00002-of-00003.gguf', 'part-00003-of-00003.gguf', 'mmproj.gguf'):
            path = self.original_model / name
            path.write_bytes(('small fixture ' + name).encode())
            model_rows.append(dict(path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for name in ('index.txt', 'native_experts.txt', 'dense.bin', 'compat-bf16.json', 'tokenizer/vocab.json',
                     'tokenizer/merges.txt', 'tokenizer/token_type.json', 'tokenizer/chat_template.jinja', 'tokenizer/tokenizer.json'):
            path = self.original_pack / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('small derived fixture ' + name).encode())
            pack_rows.append(dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        spec = importlib.util.spec_from_file_location('maya_relocation_fixture_tool', Path(__file__).parents[1] / 'relocate-verified-files.py')
        relocation = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = relocation
        spec.loader.exec_module(relocation)
        config = relocation.Config(self.original_root, self.original_model, self.runtime_root, self.runtime_model,
                                   self.receipts, self.receipts / 'model', include_files=tuple(Path(row['path']).name for row in model_rows),
                                   minimum_free_bytes=0, progress_bytes=1024)
        relocation.Relocator(config).execute()
        relocation.Relocator(relocation.Config(self.original_root, self.original_pack, self.runtime_root, self.runtime_pack,
                             self.receipts, self.receipts / 'pack', minimum_free_bytes=0, progress_bytes=1024)).execute()
        self.model_receipt = self.receipts / 'model/receipt.json'
        self.pack_receipt = self.receipts / 'pack/receipt.json'
        self.manifest = self.root / 'original-model.json'
        self.manifest.write_text(json.dumps(dict(files=model_rows)), encoding='utf-8')
        self.evidence = dict(exit_code=0, original_checkpoint_stat_unchanged=True, source_unchanged=True,
            generated_files=pack_rows, model_stat_after=[{key: row[key] for key in ('path', 'bytes', 'mtime_ns')} for row in model_rows[:3]],
            original_checkpoint_identity=dict(path=str(self.manifest), sha256=mod.bench.sha(self.manifest)))
        self.evidence_path = self.root / 'original-pack.json'
        self.evidence_path.write_text(json.dumps(self.evidence), encoding='utf-8')
        executable = self.root / 'not-an-executable.txt'
        executable.write_text('Only hash this small text; never launch it.', encoding='utf-8')
        direct = self.root / 'original-direct.json'
        direct.write_text(json.dumps(dict(quality=[dict(name='extraction', parser='production', eos=True,
             prompt=mod.fixed_tasks.PROMPTS['extraction'], generated_tokens=[1, 9])])), encoding='utf-8')
        self.plan = dict(source_path=str(self.original_root), source_revision='fixture-pin', exe=str(executable),
            exe_sha256=mod.bench.sha(executable), pack=str(self.runtime_pack),
            pack_evidence=str(self.evidence_path), pack_evidence_sha256=mod.bench.sha(self.evidence_path),
            direct_report=str(direct), direct_report_sha256=mod.bench.sha(direct), case='extraction', eos_ids=[9],
            relocation=dict(schema='maya-verified-pack-relocation.v1',
                model_receipt=dict(path=str(self.model_receipt), sha256=mod.bench.sha(self.model_receipt)),
                pack_receipt=dict(path=str(self.pack_receipt), sha256=mod.bench.sha(self.pack_receipt))))

    def rewrite_receipt(self, role, mutation):
        path = Path(self.plan['relocation'][role]['path'])
        data = json.loads(path.read_text())
        mutation(data)
        path.write_text(json.dumps(data), encoding='utf-8')
        self.plan['relocation'][role]['sha256'] = mod.bench.sha(path)

    def verify(self, plan=None):
        provenance = {}
        with patch.object(mod.subprocess, 'check_output', side_effect=['fixture-pin\n', '']):
            reference, files = mod.verify_inputs(plan or self.plan, runtime_evidence=provenance)
        return reference, files, provenance

    def test_real_four_file_and_nine_file_receipts_pass_public_verify_inputs(self):
        original_sha = mod.bench.sha
        reads = []
        def trace(path):
            reads.append(Path(path))
            return original_sha(path)
        with patch.object(mod.bench, 'sha', trace):
            reference, files, provenance = self.verify()
        self.assertEqual('extraction', reference['name'])
        self.assertEqual(9, len(files))
        self.assertEqual(4, len(provenance['model_files']))
        self.assertTrue(all(Path(row['path']).is_relative_to(self.runtime_pack) for row in files))
        self.assertTrue(all(Path(row['path']) in reads for row in files))
        self.assertFalse(any(path.suffix == '.gguf' for path in reads))
        self.assertEqual(self.verify()[1:], (files, provenance))

    def test_original_plan_strict_behavior_is_preserved_and_mixed_pack_rejected(self):
        old = copy.deepcopy(self.plan)
        del old['relocation']
        with self.assertRaisesRegex(ValueError, 'Pack path differs'):
            self.verify(old)
        old['pack'] = str(self.original_pack)
        _, files, provenance = self.verify(old)
        self.assertEqual(self.evidence['generated_files'], files)
        self.assertEqual({}, provenance)

    def test_uncompleted_or_retired_receipt_never_qualifies(self):
        for status in ('copying', 'verified', 'failed', 'complete'):
            with self.subTest(status=status):
                original = self.pack_receipt.read_text()
                self.rewrite_receipt('pack_receipt', lambda row: row.update(status=status))
                with self.assertRaisesRegex(ValueError, 'completed verified copy'):
                    self.verify()
                self.pack_receipt.write_text(original, encoding='utf-8')
                self.plan['relocation']['pack_receipt']['sha256'] = mod.bench.sha(self.pack_receipt)
        self.rewrite_receipt('model_receipt', lambda row: row['config'].update(retire_source=True))
        with self.assertRaisesRegex(ValueError, 'copy-only'):
            self.verify()

    def test_wrong_named_root_and_mixed_original_runtime_paths_refuse(self):
        self.rewrite_receipt('pack_receipt', lambda row: row['config'].update(destination_root=str(self.runtime_pack)))
        with self.assertRaisesRegex(ValueError, 'named roots'):
            self.verify()

    def test_original_pack_cannot_be_mixed_with_relocated_shards(self):
        self.plan['pack'] = str(self.original_pack)
        with self.assertRaisesRegex(ValueError, 'source/destination'):
            self.verify()

    def test_missing_ninth_pack_file_and_unrecorded_experts_override_refuse(self):
        self.rewrite_receipt('pack_receipt', lambda row: row['files'].pop('tokenizer/tokenizer.json'))
        with self.assertRaises(ValueError):
            self.verify()

    def test_unrecorded_experts_bin_is_rejected_even_with_valid_receipts(self):
        (self.runtime_pack / 'experts.bin').write_bytes(b'foreign expert override')
        with self.assertRaisesRegex(ValueError, 'Unrecorded or missing'):
            self.verify()

    def test_pinned_receipt_file_sha_change_and_content_sha_mismatch_refuse(self):
        self.plan['relocation']['pack_receipt']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Pinned relocation receipt'):
            self.verify()
        self.plan['relocation']['pack_receipt']['sha256'] = mod.bench.sha(self.pack_receipt)
        self.rewrite_receipt('pack_receipt', lambda row: row['files']['dense.bin'].update(sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
            self.verify()

    def test_actual_same_size_pack_corruption_cannot_reuse_receipt(self):
        path = self.runtime_pack / 'dense.bin'
        before = path.stat()
        path.write_bytes(b'X' * before.st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaises(ValueError):
            self.verify()

    def test_actual_destination_shard_metadata_change_and_source_snapshot_mismatch_refuse(self):
        path = self.runtime_model / 'part-00002-of-00003.gguf'
        info = path.stat()
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000))
        with self.assertRaisesRegex(ValueError, 'metadata changed'):
            self.verify()

    def test_source_snapshot_cannot_rebind_a_different_original_timestamp(self):
        def mutate(receipt):
            row = receipt['files']['part-00002-of-00003.gguf']['source_snapshot']
            row['mtime_ns'] += 1000000
        self.rewrite_receipt('model_receipt', mutate)
        with self.assertRaisesRegex(ValueError, 'source timestamp'):
            self.verify()

    def test_partial_pack_or_unselected_checkpoint_receipt_is_refused(self):
        self.rewrite_receipt('pack_receipt', lambda row: row['config'].update(include_files=list(row['files'])))
        with self.assertRaisesRegex(ValueError, 'entire unfiltered pack'):
            self.verify()

    def test_duplicate_include_paths_unverified_row_and_traversal_refuse(self):
        original = self.model_receipt.read_text()
        mutations = [lambda row: row['config']['include_files'].append(row['config']['include_files'][0]),
                     lambda row: row['files'][next(iter(row['files']))].update(state='publishing'),
                     lambda row: row['files'].update({'../outside.gguf': row['files'].pop(next(iter(row['files'])))})]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.model_receipt.write_text(original, encoding='utf-8')
                self.rewrite_receipt('model_receipt', mutation)
                with self.assertRaises(ValueError):
                    self.verify()


if __name__ == '__main__':
    unittest.main()
