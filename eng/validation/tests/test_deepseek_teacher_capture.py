"""Model-free native-boundary/call-schedule tests. No CDLL/model/GPU is opened."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('teacher_capture', Path(__file__).parents[1] / 'capture-deepseek-teacher-logits.py')
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


def fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    native = 'a' * 64
    load = {'backend': 'CUDA', 'n_gpu': 7, 'n_ctx': 65536, 'n_ubatch': 2, 'n_threads': 32, 'n_cpu_moe': 12}
    model = {'directory': '/models', 'manifest_sha256': 'd' * 64,
             'files': [{'path': 'model.gguf', 'size': 123, 'expected_sha256': 'e' * 64}]}
    plan = {'variants': {name: {'expected_native_sha256': native, 'native_load': load,
             'environment': {'TS_DSV41_TP': str(degree), 'TS_DSV4_FA': '1'},
             'requested_expert_tp_ranks': degree} for name, degree in [('non-tp', 0), ('expert-tp7', 7)]},
            'model': model, 'request_count': 1, 'requests': [{'id': 'request0'}], 'protocol': {'context': 65536},
            'gates': {'existing_fixture_allclose_diagnostic': {'atol': 2e-5, 'rtol': 2e-5}}}
    identity = {'native_sha256': native, 'native_source_sha256': {'owned.cpp': 'f' * 64},
                'exporter_source_sha256': {'TokenExporter.cs': '1' * 64}, 'exporter_assemblies_sha256': {'Exporter.dll': '2' * 64},
                'token_export_sha256': '3' * 64, 'capture_program_sha256': '4' * 64,
                'model_manifest_sha256': model['manifest_sha256'],
                'checkpoint_files': {'/models/model.gguf': {'bytes': 123, 'sha256': 'e' * 64}},
                'engram_files': {'/models/engram.bin': {'bytes': 123, 'sha256': '5' * 64}},
                'tokenizer': {'first_gguf_sha256': 'e' * 64, 'vocab_size': 4, 'eos_ids': [3]}}
    rows, history = [], b''
    for index, (tokens, stage, forced) in enumerate([([0, 1], 'prompt', None), ([2], 'prompt', 2), ([2], 'continuation', 3)]):
        raw = capture.token_bytes(tokens)
        before = len(history) // 4
        history += raw
        name = f'{index}.i32'
        (root / name).write_bytes(raw)
        rows.append({'row_id': f'request0/primary/{index}', 'request_id': 'request0', 'phase': 'primary',
                     'slot_id': 0, 'call_index': index, 'position_before': before, 'input_token_count': len(tokens),
                     'position_after': len(history) // 4, 'predicted_token_position': len(history) // 4,
                     'cumulative_input_i32_sha256': capture.sha(history), 'call_input_i32_sha256': capture.sha(raw),
                     'next_forced_token_id': forced, 'vocab_size': 4, 'stage': stage, 'call_input_file': name})
    prompt = capture.token_bytes([0, 1, 2])
    (root / '000.final.i32').write_bytes(prompt)
    token_export = {'status': 'tokens-exported', 'release_qualified': False, 'requests': [
        {'id': 'request0', 'prefix': '000', 'TokensI32Sha256': capture.sha(prompt), 'prompt_tokens': 3, 'RemovedMessages': 0}]}
    token_export.update(plan_sha256='b' * 64, tokenizer={'VocabSize': 4, 'EosTokenIds': [3]},
                        tokenizer_source={'original_first_shard_sha256': 'e' * 64})
    for name in ('before', 'after'):
        token_export[name] = {'native_mappings': [], 'source_sha256': identity['exporter_source_sha256'],
                              'assembly_sha256': identity['exporter_assemblies_sha256']}
    continuation = {'schema_version': 1, 'status': 'complete', 'source_variant': 'non-tp', 'maximum_tokens': 64,
                    'native_sha256': native, 'token_export_sha256': identity['token_export_sha256'], 'source_capture_sha256': '6' * 64,
                    'requests': [{'id': 'request0', 'tokens': [2, 3], 'stop_reason': 'eos'}]}
    schedule = {'schema_version': 1, 'teacher_plan_sha256': 'b' * 64, 'dtype': 'float32-le',
                'shared_identity': identity, 'rows': rows, 'coverage': 'complete-primary', 'capture_scope': capture.SUPPORTED_SCOPE}
    observation = {'pid': 123, 'start_ticks': 456, 'boot_id': 'scripted-offline',
                   'mapped_native_libraries': {'/app/libGgmlOps.so': native},
                   **{key: identity[key] for key in ('native_source_sha256', 'checkpoint_files', 'engram_files', 'token_export_sha256')}}
    report = {'schema_version': 1, 'schedule_sha256': 'c' * 64, 'shared_identity': identity, 'variant': 'non-tp',
              'native_load': load, 'environment': plan['variants']['non-tp']['environment'], 'native_library_path': '/app/libGgmlOps.so',
              'placement': {'effective_tp_ranks': 0, 'effective_gpu_count': 7, 'effective_cpu_moe_layers': 12,
                            'layer_placement_evidence': ['SCRIPTED'], 'rank_shard_evidence': []},
              'observations': {'before': copy.deepcopy(observation), 'after': copy.deepcopy(observation)}, 'scripted_fixture': True}
    return plan, schedule, token_export, continuation, report


class FakeNative:
    def __init__(self, failure=None):
        self.position = 99
        self.calls = []
        self.failure = failure
    def vocab(self): return 3 if self.failure == 'vocab' else 4
    def context(self): return 32768 if self.failure == 'context' else 65536
    def past(self): return self.position
    def reset(self):
        self.calls.append(('reset',))
        self.position = 7 if self.failure == 'reset' else 0
    def forward(self, raw, output):
        tokens = list(struct.unpack('<' + 'i' * (len(raw) // 4), raw))
        self.calls.append(('forward', tokens))
        if self.failure == 'exception':
            output[0] = 5
            raise RuntimeError('scripted ctypes boundary failure')
        for i in range(1 if self.failure in ('status', 'partial') else 4):
            # Greedy choice0 intentionally differs from forced token2. Capture
            # must NEVER use this choice to mutate the independent input stream.
            output[i] = (8, -0.0, 2, -3)[i]
        self.position += len(tokens) + (1 if self.failure == 'position' else 0)
        return -5 if self.failure == 'status' else 0


class TeacherCaptureTests(unittest.TestCase):
    def test_exporter_actual_identity_eos_source_and_native_guard_cannot_be_redeclared(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, schedule, exported, _, _ = fixture(Path(temp))
            for mutate in (lambda e: e['tokenizer'].update(EosTokenIds=[2]),
                           lambda e: e.update(plan_sha256='9' * 64),
                           lambda e: e['tokenizer_source'].update(original_first_shard_sha256='9' * 64),
                           lambda e: e['after']['native_mappings'].append('GgmlOps.dll'),
                           lambda e: e['before'].update(source_sha256={'different.cs': '8' * 64}),
                           lambda e: e['after'].update(assembly_sha256={'different.dll': '7' * 64})):
                invalid = copy.deepcopy(exported)
                mutate(invalid)
                with self.assertRaises(ValueError):
                    capture.validate_export_identity(plan, 'b' * 64, schedule['shared_identity'], invalid)

    def test_failure_still_audits_before_free_and_preserves_all_three_errors(self):
        for audit_failure in (False, True):
            events = []
            def audit():
                events.append('audit')
                if audit_failure:
                    raise RuntimeError('secondary source audit failed')
                return {'latest': True}
            def free():
                events.append('free')
                raise RuntimeError('tertiary cleanup failed')
            report = {'status': 'failed', 'error': 'original Forward failed', 'observations': {}}
            capture.audit_and_free(types.SimpleNamespace(free=free), types.SimpleNamespace(audit=audit), report, {}, 'x', {})
            self.assertEqual(events, ['audit', 'free'])
            self.assertEqual(report['error'], 'original Forward failed')
            self.assertEqual(report['status'], 'failed')
            self.assertIn('tertiary', report['cleanup_error'])
            if audit_failure:
                self.assertIn('secondary', report['after_audit_error'])
            else:
                self.assertEqual(report['observations']['after'], {'latest': True})

    def test_successful_rows_cannot_hide_failed_final_identity(self):
        before = {'mapped_native_libraries': {'lib': 'a'}, 'computation_environment': {'TS_X': '1'}}
        after = copy.deepcopy(before)
        after['computation_environment']['TS_X'] = '0'
        native = mock.Mock()
        report = {'status': 'complete', 'observations': {'before': before}}
        capture.audit_and_free(native, types.SimpleNamespace(audit=lambda: after), report, {}, 'x', {})
        self.assertEqual(report['status'], 'failed')
        self.assertIn('Computation environment changed', report['after_audit_error'])
        native.free.assert_called_once_with()

    def test_comparator_executes_the_checked_bytes_without_reopen_or_pyc(self):
        with mock.patch.object(capture, 'read_pinned', return_value=b'from_checked_bytes = 42\n') as read:
            module = capture.load_comparator()
            self.assertEqual(module.from_checked_bytes, 42)
            read.assert_called_once()

    def test_full_row_bytes_positions_forcing_and_comparator_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, schedule, exported, continuation, report = fixture(root / 'input')
            rows = capture.validate_serial_schedule(plan, 'b' * 64, schedule, root / 'input', exported, root / 'input', continuation)
            native = FakeNative()
            result = capture.capture_rows(native, rows, root / 'input', root / 'out', report)
            self.assertEqual(result['status'], 'complete')
            self.assertFalse(result['release_qualified'])
            self.assertEqual(native.calls, [('reset',), ('forward', [0, 1]), ('forward', [2]), ('forward', [2])])
            expected = struct.pack('<4f', 8, -0.0, 2, -3)
            self.assertEqual((root / 'out/logits.f32').read_bytes(), expected * 3)
            self.assertEqual([r['byte_offset'] for r in result['rows']], [0, 16, 32])
            compare = capture.load_comparator()
            compare.validate_capture(result, schedule, 'c' * 64, plan)
            compare.payload_layout(root / 'out', result['rows'])
            for row in result['rows']:
                _, raw = compare.read_row(root / 'out', row)
                self.assertEqual(raw, expected)

    def test_failed_forward_partial_write_and_exception_preserve_complete_raw_row(self):
        for failure in ('status', 'partial', 'exception', 'position'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                _, schedule, _, _, report = fixture(root / 'input')
                with self.assertRaises(ValueError):
                    capture.capture_rows(FakeNative(failure), schedule['rows'], root / 'input', root / 'out', report)
                saved = json.loads((root / 'out/capture.json').read_text())
                self.assertEqual(saved['status'], 'failed')
                self.assertFalse(saved['release_qualified'])
                self.assertEqual(len(saved['rows']), 1)
                self.assertEqual((root / 'out/logits.f32').stat().st_size, 16)
                coordinates = json.loads((root / 'out/nonfinite-coordinates.jsonl').read_text())
                self.assertEqual(coordinates['nonfinite_coordinates'], [] if failure == 'position' else [1, 2, 3])

    def test_reset_vocab_and_context_fail_before_any_forward(self):
        for failure in ('reset', 'vocab', 'context'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                _, schedule, _, _, report = fixture(root / 'input')
                native = FakeNative(failure)
                with self.assertRaises(ValueError):
                    capture.capture_rows(native, schedule['rows'], root / 'input', root / 'out', report)
                self.assertFalse(any(call[0] == 'forward' for call in native.calls))
                self.assertEqual(report['rows'], [])

    def test_invalid_scope_positions_tokens_or_partial_schedule_rejected(self):
        changes = {
            'slot': lambda s, e, c: s['rows'][0].update(slot_id=1),
            'phase': lambda s, e, c: s['rows'][0].update(phase='boundary'),
            'context': lambda s, e, c: s['rows'][0].update(position_after=65537),
            'prediction': lambda s, e, c: s['rows'][0].update(predicted_token_position=1),
            'missing': lambda s, e, c: s['rows'].pop(),
            'call': lambda s, e, c: s['rows'][1].update(call_index=7),
            'history': lambda s, e, c: s['rows'][1].update(cumulative_input_i32_sha256='0' * 64),
            'forced': lambda s, e, c: s['rows'][1].update(next_forced_token_id=0),
            'candidate-teacher': lambda s, e, c: c.update(source_variant='expert-tp7'),
            'compacted': lambda s, e, c: e['requests'][0].update(RemovedMessages=1),
            'unfrozen': lambda s, e, c: c.update(source_capture_sha256=None),
        }
        for name, mutate in changes.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                plan, schedule, exported, continuation, _ = fixture(root)
                mutate(schedule, exported, continuation)
                with self.assertRaises((ValueError, KeyError)):
                    capture.validate_serial_schedule(plan, 'b' * 64, schedule, root, exported, root, continuation)

    def test_hash_and_vocabulary_guards_and_input_changes_after_preflight(self):
        for kind in ('bytes', 'token-id'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                plan, schedule, exported, continuation, report = fixture(root / 'input')
                capture.validate_serial_schedule(plan, 'b' * 64, schedule, root / 'input', exported, root / 'input', continuation)
                bad = capture.token_bytes([0, 4])
                (root / 'input/0.i32').write_bytes(bad)
                if kind == 'token-id': schedule['rows'][0]['call_input_i32_sha256'] = capture.sha(bad)
                native = FakeNative()
                with self.assertRaises(ValueError):
                    capture.capture_rows(native, schedule['rows'], root / 'input', root / 'out', report)
                self.assertFalse(any(call[0] == 'forward' for call in native.calls))

    def test_last_eos_prediction_is_not_fed_back(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, schedule, exported, continuation, _ = fixture(root)
            row = copy.deepcopy(schedule['rows'][-1])
            raw = capture.token_bytes([3])
            (root / 'eos.i32').write_bytes(raw)
            row.update(row_id='extra-eos', call_index=3, position_before=4, position_after=5, predicted_token_position=5,
                       call_input_file='eos.i32', call_input_i32_sha256=capture.sha(raw),
                       cumulative_input_i32_sha256=capture.sha(capture.token_bytes([0, 1, 2, 2, 3])), next_forced_token_id=None)
            schedule['rows'].append(row)
            with self.assertRaisesRegex(ValueError, 'feeds EOS'):
                capture.validate_serial_schedule(plan, 'b' * 64, schedule, root, exported, root, continuation)

    def test_64_token_cap_preserves_final_prediction_without_an_extra_forward(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, schedule, exported, continuation, report = fixture(root / 'input')
            continuation['requests'][0].update(tokens=[1] * 64, stop_reason='maximum_tokens')
            schedule['rows'] = schedule['rows'][:2]
            schedule['rows'][-1]['next_forced_token_id'] = 1
            history = capture.token_bytes([0, 1, 2])
            for index in range(63):
                raw = capture.token_bytes([1])
                name = f'cap-{index}.i32'
                (root / 'input' / name).write_bytes(raw)
                before = len(history) // 4
                history += raw
                schedule['rows'].append({'row_id': f'request0/primary/{index+2}', 'request_id': 'request0', 'phase': 'primary',
                    'slot_id': 0, 'call_index': index + 2, 'position_before': before, 'position_after': before + 1,
                    'predicted_token_position': before + 1, 'input_token_count': 1, 'vocab_size': 4,
                    'cumulative_input_i32_sha256': capture.sha(history), 'call_input_i32_sha256': capture.sha(raw),
                    'call_input_file': name, 'stage': 'continuation', 'next_forced_token_id': 1})
            rows = capture.validate_serial_schedule(plan, 'b' * 64, schedule, root / 'input', exported, root / 'input', continuation)
            native = FakeNative()
            capture.capture_rows(native, rows, root / 'input', root / 'out', report)
            self.assertEqual(native.position, 3 + 63)
            self.assertEqual(len(report['rows']), 2 + 63)
            self.assertEqual(sum(row['next_forced_token_id'] is not None for row in report['rows']), 64)

    def test_each_request_reset_and_its_independent_history_start_at_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, schedule, _, _, report = fixture(root / 'input')
            second = copy.deepcopy(schedule['rows'])
            for row in second:
                row['request_id'] = 'second'
                row['row_id'] = row['row_id'].replace('request0', 'second')
            native = FakeNative()
            capture.capture_rows(native, schedule['rows'] + second, root / 'input', root / 'out', report)
            self.assertEqual(sum(c[0] == 'reset' for c in native.calls), 2)
            self.assertEqual(len(report['rows']), 6)

    def test_explicit_abi_binding_has_no_silent_checked_reset_fallback(self):
        class Function:
            def __call__(self, *args): return 0
        names = ['LoadModel', 'Free', 'VocabSize', 'CtxSize', 'NPast', 'Forward', 'Reset']
        dll = types.SimpleNamespace(**{'TSGgml_Dsv4' + name: Function() for name in names})
        with mock.patch.object(capture.ctypes, 'CDLL', return_value=dll):
            native = capture.Native('/not-loaded.so', 'v5-void')
            self.assertEqual(len(native._forward.argtypes), 4)
            self.assertIsNone(native._reset.restype)
            with self.assertRaises(AttributeError): capture.Native('/not-loaded.so', 'v6-checked')
            with self.assertRaises(ValueError): capture.Native('/not-loaded.so', 'auto')


class PlacementTests(unittest.TestCase):
    @staticmethod
    def log(tp):
        lines = ['[dsv4] compute devices initialized: 7 CUDA',
                 '[dsv4] routed-expert CPU offload: 12 of 40 layer(s); 28 layer(s) on GPUs']
        if tp:
            lines.append('[dsv41] routed-MoE tensor parallelism: 7 ranks, sharded gate/up/down weights; attention uses layer placement')
        for device in range(7):
            if tp: lines.append(f'[dsv41]   rank {device}: 31.160 GiB of routed-expert tensor shards')
            first, last = device * 6, min(39, device * 6 + 5)
            lines.append(f'[dsv4]   device {device}: layers {first}..{last} ({last-first+1}), 12.0 GiB free after load')
        return '\n'.join(lines)

    def test_actual_placement_parser_requires_all_layers_devices_and_rank_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _, _, _, _ = fixture(Path(temp))
            for variant, tp in [('non-tp', False), ('expert-tp7', True)]:
                result = capture.parse_placement(self.log(tp), plan['variants'][variant], 40)
                self.assertEqual(result['effective_tp_ranks'], 7 if tp else 0)
                self.assertEqual(len(result['layer_placement_evidence']), 7)
            for bad in [self.log(False), self.log(True).replace('12 of 40', '11 of 40'),
                        self.log(True).replace('rank 6:', 'rank 5:'), self.log(True).replace('layers 36..39 (4)', 'layers 35..39 (5)'),
                        self.log(True).replace('31.160 GiB', '0.000 GiB')]:
                with self.assertRaises(ValueError): capture.parse_placement(bad, plan['variants']['expert-tp7'], 40)


class ObserverTests(unittest.TestCase):
    def test_exact_files_mapping_and_immutable_stats_are_required_at_each_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {name: root / name for name in ('libGgmlOps.so', 'owned.cpp', 'export.json', 'model.gguf', 'engram.bin')}
            for name, path in paths.items():
                path.write_bytes(name.encode())
            identity = {'native_sha256': capture.file_sha(paths['libGgmlOps.so']),
                        'native_source_sha256': {'owned.cpp': capture.file_sha(paths['owned.cpp'])},
                        'token_export_sha256': capture.file_sha(paths['export.json']),
                        'capture_program_sha256': capture.file_sha(capture.__file__), 'model_manifest_sha256': 'd' * 64}
            for category, name in [('checkpoint_files', 'model.gguf'), ('engram_files', 'engram.bin')]:
                identity[category] = {str(paths[name]): {'bytes': paths[name].stat().st_size, 'sha256': capture.file_sha(paths[name])}}
            attestation = {'verification': 'publisher-hash-reuse-with-immutable-stat-attestation', 'model_manifest_sha256': 'd' * 64}
            for category in ('checkpoint_files', 'engram_files'):
                attestation[category] = {name: {**item, 'stat': capture.stat_identity(name)} for name, item in identity[category].items()}
            attestation_path = root / 'attestation.json'
            attestation_path.write_text(json.dumps(attestation), encoding='utf-8')
            identity['model_attestation_sha256'] = capture.file_sha(attestation_path)
            config = {'native_library': str(paths['libGgmlOps.so']), 'native_source_root': str(root),
                      'token_export_path': str(paths['export.json']), 'model_attestation_path': str(attestation_path)}
            original_read = Path.read_text
            mappings = ['1000-2000 r--p 00000000 00:00 0 ' + str(paths['libGgmlOps.so'].resolve())]
            def proc_read(path, *args, **kwargs):
                name = path.as_posix()
                if name == '/proc/self/maps': return '\n'.join(mappings)
                if name == '/proc/self/stat': return '1 (scripted python) S ' + ' '.join(['0'] * 18 + ['456'])
                if name == '/proc/sys/kernel/random/boot_id': return 'offline-boot'
                return original_read(path, *args, **kwargs)
            observer = capture.Observer(config, {'shared_identity': identity})
            with mock.patch.object(Path, 'read_text', proc_read):
                first = observer.audit()
                self.assertEqual(first['start_ticks'], 456)
                self.assertEqual(first['mapped_native_libraries'], {str(paths['libGgmlOps.so'].resolve()): identity['native_sha256']})
                mappings.append(mappings[0] + ' (deleted)')
                with self.assertRaisesRegex(ValueError, 'Deleted native mapping'): observer.audit()
                mappings.pop()
                paths['engram.bin'].write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Model file changed'): observer.audit()
                paths['owned.cpp'].write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Native source changed'): observer.audit()


class CliTests(unittest.TestCase):
    def fixture(self, root):
        plan, schedule, exported, continuation, report = fixture(root)
        def write(name, value):
            path = root / name
            path.write_text(json.dumps(value), encoding='utf-8')
            return str(path), capture.file_sha(path)
        native = root / 'libGgmlOps.so'
        native.write_bytes(b'SCRIPTED-UNLOADABLE-NATIVE')
        identity = schedule['shared_identity']
        identity.update(native_sha256=capture.file_sha(native), capture_program_sha256=capture.file_sha(capture.__file__), driver_environment={})
        for variant in plan['variants'].values(): variant['expected_native_sha256'] = identity['native_sha256']
        plan_path, plan_sha = write('plan.json', plan)
        exported['plan_sha256'] = plan_sha
        export_path, identity['token_export_sha256'] = write('export.json', exported)
        attest_path, identity['model_attestation_sha256'] = write('attestation.json', {'scripted': True})
        failed_path, failed_sha = write('failed.json', {'status': 'failed', 'native_sha256': identity['native_sha256'],
            'source_sha256': identity['native_source_sha256'], 'steps': [{'name': 'native', 'exit_code': 0}, {'name': 'tp7-checkpoint-shape', 'exit_code': 1}]})
        report.update(status='complete', phase='independent-baseline-continuation', producer_sha256='8' * 64,
                      native_library_path=str(native.resolve()), original_failed_build_sha256=failed_sha, release_qualified=False)
        for observation in report['observations'].values():
            observation.update(mapped_native_libraries={str(native.resolve()): identity['native_sha256']},
                               token_export_sha256=identity['token_export_sha256'], computation_environment={})
        report['rows'] = [{**row, 'native_status': 0, 'logits_sha256': '9' * 64} for row in schedule['rows']]
        baseline_path, baseline_sha = write('baseline.json', report)
        continuation.update(native_sha256=identity['native_sha256'], token_export_sha256=identity['token_export_sha256'],
                            source_capture_sha256=baseline_sha, producer_sha256='8' * 64)
        continuation_path, continuation_sha = write('continuation.json', continuation)
        schedule.update(teacher_plan_sha256=plan_sha, original_failed_build_sha256=failed_sha, baseline_continuation_sha256=continuation_sha)
        schedule_path, schedule_sha = write('schedule.json', schedule)
        config = {'scope': capture.SUPPORTED_SCOPE, 'release_qualified': False, 'plan_path': plan_path, 'plan_sha256': plan_sha,
            'schedule_path': schedule_path, 'schedule_sha256': schedule_sha, 'native_library': str(native), 'native_source_root': str(root),
            'variant': 'non-tp', 'original_failed_build_path': failed_path, 'original_failed_build_sha256': failed_sha,
            'token_export_path': export_path, 'continuation_path': continuation_path, 'baseline_capture_path': baseline_path,
            'model_attestation_path': attest_path, 'reset_api': 'v5-void', 'model_layers': 40}
        config_path, config_sha = write('config.json', config)
        return config_path, config_sha, report

    def test_complete_cli_uses_explicit_pins_and_failed_row_still_audits_before_free(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config, pin, source_report = self.fixture(root / 'inputs')
                output = root / 'out'
                events = []
                native = FakeNative('status' if failure else None)
                native.load = lambda *_: (output / 'native.log').write_text(PlacementTests.log(False))
                native.free = lambda: events.append('free')
                observation = copy.deepcopy(source_report['observations']['before'])
                count = 0
                def audit():
                    nonlocal count
                    count += 1
                    events.append('audit')
                    if failure and count == 2: raise RuntimeError('secondary final audit')
                    return copy.deepcopy(observation)
                observer = types.SimpleNamespace(audit=audit)
                argv = ['capture.py', '--config', config, '--config-sha256', pin, '--output', str(output)]
                with mock.patch.object(capture.sys, 'argv', argv), mock.patch.object(capture.sys, 'platform', 'linux'), \
                        mock.patch.dict(capture.os.environ, {}, clear=True), mock.patch.object(capture, 'Native', return_value=native), \
                        mock.patch.object(capture, 'Observer', return_value=observer), contextlib.redirect_stdout(io.StringIO()):
                    if failure:
                        with self.assertRaisesRegex(ValueError, 'Forward failed'): capture.main()
                    else:
                        capture.main()
                result = json.loads((output / 'lifecycle.json').read_text())
                self.assertEqual(events, ['audit', 'audit', 'free'])
                self.assertFalse(result['release_qualified'])
                if failure:
                    self.assertEqual(result['status'], 'failed')
                    self.assertIn('Forward failed', result['error'])
                    self.assertIn('secondary final audit', result['after_audit_error'])
                    self.assertEqual((output / 'rows/logits.f32').stat().st_size, 16)
                else:
                    self.assertEqual(result['status'], 'complete')
                    self.assertEqual(len(result['rows']), 3)
                    self.assertTrue((output / 'baseline-capture.json').exists())

    def test_cli_rejects_bad_config_before_constructing_native(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _, _ = self.fixture(root / 'inputs')
            with mock.patch.object(capture.sys, 'argv', ['capture.py', '--config', config, '--config-sha256', '0' * 64, '--output', str(root / 'out')]), \
                    mock.patch.object(capture.sys, 'platform', 'linux'), mock.patch.object(capture, 'Native') as native:
                with self.assertRaisesRegex(ValueError, 'Pinned artifact changed'): capture.main()
                native.assert_not_called()
                self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
