"""Pure gates only: no native import, device, model, build or large file read."""
import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'qualify-host-moe-components.py'
spec = importlib.util.spec_from_file_location('host_moe_components', path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def phase_line(mode, elements=None):
    budget = 467 if mode == 0 else 917
    if elements is None:
        elements = tool.PHASE_ROWS[mode] * tool.PHASE_COLUMNS
    return (f'PASS: actual phase coexist={mode} minimum=450 budget={budget} all-wide=900; '
            f'{elements} complete finite CUDA operator floats byte-exact, PP128/256/384/512/1024/N3/group switch, '
            'min-1 nonmutation, reshape faults, read interruption, trim/resume/drop/end. '
            'Not whole-model logits or asynchronous cancellation qualification.\n')


class HostMoeComponentTests(unittest.TestCase):
    def test_identical_derived_driver_needs_no_overlay_but_overlay_requires_component(self):
        source = {'TensorSharp.GGML.Native/tests/example.cpp': 'a' * 64}
        self.assertEqual({}, tool.resolve_test_overlays(Path('fresh'), None, source, 'b' * 64))
        self.assertEqual({}, tool.resolve_test_overlays(None, None, source, 'b' * 64))
        overlay = {'schema': 'tensorsharp.native-component-test-overlay.v1',
                   'base_build_receipt_sha256': 'b' * 64, 'files': {next(iter(source)): {
                       'original_sha256': 'a' * 64, 'replacement_sha256': 'c' * 64}}}
        with self.assertRaisesRegex(ValueError, 'requires an isolated'):
            tool.resolve_test_overlays(None, overlay, source, 'b' * 64)
        self.assertEqual(overlay['files'], tool.resolve_test_overlays(Path('fresh'), overlay, source, 'b' * 64))

    def test_optional_overlay_keeps_strict_receipt_source_scope_and_hash_requirements(self):
        relative = 'TensorSharp.GGML.Native/tests/example.cpp'
        source = {relative: 'a' * 64, 'TensorSharp.GGML.Native/cache.cpp': 'd' * 64}
        def overlay(name=relative, original='a' * 64, replacement='c' * 64):
            return {'schema': 'tensorsharp.native-component-test-overlay.v1',
                    'base_build_receipt_sha256': 'b' * 64, 'files': {name: {
                        'original_sha256': original, 'replacement_sha256': replacement}}}
        invalid = [overlay(original='e' * 64), overlay(replacement='a' * 64), overlay(replacement='c'),
                   overlay(name='TensorSharp.GGML.Native/cache.cpp', original='d' * 64),
                   overlay(name='TensorSharp.GGML.Native/tests/../example.cpp'),
                   overlay(name='TensorSharp.GGML.Native/tests/missing.cpp'),
                   {**overlay(), 'schema': 'other'}, {**overlay(), 'files': {}},
                   {**overlay(), 'base_build_receipt_sha256': 'e' * 64}]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                tool.resolve_test_overlays(Path('fresh'), item, source, 'b' * 64)

    def test_identical_derived_driver_copies_exact_pinned_tree_and_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = object.__new__(tool.Runner)
            root = Path(temporary); runner.build = root / 'build'; runner.component = root / 'derived'
            runner.driver = runner.component / 'driver'; runner.repo = root / 'repo'; runner.overlays = {}
            originals = {'TensorSharp.GGML.Native/CMakeLists.txt': b'formal cmake\n',
                         'TensorSharp.GGML.Native/tests/example.cpp': b'complete fixture\n'}
            runner.source = {name: hashlib.sha256(raw).hexdigest() for name, raw in originals.items()}
            generated = {'driver/CMakeLists.txt': b'generated cmake with imported unchanged upstream\n',
                         'import-upstream.cmake': b'imported libraries\n'}
            runner.full = {'generated_driver_files': {name: hashlib.sha256(raw).hexdigest() for name, raw in generated.items()}}
            for name, raw in originals.items():
                suffix = Path(name).relative_to('TensorSharp.GGML.Native'); file = runner.build / 'driver' / suffix
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(generated.get('driver/' + suffix.as_posix(), raw))
            for name, raw in generated.items(): (runner.build / name).write_bytes(raw)
            runner.prepare_component_driver()
            self.assertEqual(generated['driver/CMakeLists.txt'], (runner.driver / 'CMakeLists.txt').read_bytes())
            self.assertEqual(originals['TensorSharp.GGML.Native/tests/example.cpp'], (runner.driver / 'tests/example.cpp').read_bytes())
            self.assertEqual(generated['import-upstream.cmake'], (runner.component / 'import-upstream.cmake').read_bytes())
            with self.assertRaises(FileExistsError): runner.prepare_component_driver()

    def test_identical_derivation_rejects_changed_frozen_source_before_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = object.__new__(tool.Runner); root = Path(temporary)
            runner.build = root / 'build'; runner.component = root / 'derived'; runner.driver = runner.component / 'driver'
            runner.repo = root / 'repo'; runner.overlays = {}
            relative = 'TensorSharp.GGML.Native/tests/example.cpp'
            runner.source = {relative: hashlib.sha256(b'original').hexdigest()}
            runner.full = {'generated_driver_files': {}}
            frozen = runner.build / 'driver/tests/example.cpp'; frozen.parent.mkdir(parents=True); frozen.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'changed before derivation'): runner.prepare_component_driver()
            self.assertFalse((runner.driver / 'tests/example.cpp').exists())

    def phase(self, text):
        case = next(c for c in tool.cases('cuda') if c['name'] == 'phase-three-groups')
        return tool.analyze_component_output(case, text)

    def test_additive_matrix_preserves_static_and_lifecycle_cases(self):
        self.assertEqual(8, len(tool.TARGETS))
        self.assertEqual(12, len(tool.cases('cpu')))
        self.assertEqual(29, len(tool.cases('cuda')))
        cases = {c['name']: c for c in tool.cases('cuda')}
        self.assertEqual(['--phase-quotas'], cases['phase-three-groups']['args'])
        self.assertEqual(['--group-quotas'], cases['measured-three-groups']['args'])
        for suffix, flags in [('q8', []), ('iq1', ['--quantization', 'iq1_m']),
                              ('iq2', ['--quantization', 'iq2_xxs'])]:
            self.assertEqual(['--split-lifecycle', *flags], cases['split-' + suffix]['args'])
            self.assertEqual(['--glm-compact-prefill', *flags], cases['compact-' + suffix]['args'])
        self.assertIn('scalar-large-file-iq1', cases)
        self.assertIn('pooled-largest-quota-iq1', cases)
        self.assertEqual(['--glm-scalar-file-window', '--direct-reads', '--quantization', 'iq1_m'], cases['scalar-buffered-under-direct-iq1']['args'])
        self.assertEqual(['--phase-quotas', '--direct-reads'], cases['phase-three-groups-direct']['args'])

    def test_complete_phase_modes_count_only_finite_exact_success_rows(self):
        actual = self.phase(phase_line(0) + phase_line(1))
        self.assertTrue(actual['passed'])
        self.assertEqual([13861, 19241], [row['operator_rows'] for row in actual['phase_modes']])
        self.assertEqual(8474112, sum(row['finite_exact_float_elements'] for row in actual['phase_modes']))
        self.assertIn('not model vocabulary logits', actual['scope'])

    def test_missing_duplicate_or_zero_phase_cannot_pass(self):
        for text in ['', phase_line(0), phase_line(1), phase_line(0) * 2,
                     phase_line(0) + phase_line(1) * 2, phase_line(0, 0) + phase_line(1)]:
            with self.subTest(text=text[:90]):
                self.assertFalse(self.phase(text)['passed'])

    def test_wrong_float_count_or_budget_cannot_pass(self):
        valid = phase_line(0) + phase_line(1)
        for text in [phase_line(0, 462081) + phase_line(1), valid.replace('budget=467', 'budget=449'),
                     valid.replace('budget=917', 'budget=918'), valid.replace('all-wide=900', 'all-wide=450')]:
            self.assertFalse(self.phase(text)['passed'])

    def test_partial_or_tolerance_based_coverage_cannot_pass(self):
        valid = phase_line(0) + phase_line(1)
        for old, new in [('byte-exact', 'within tolerance'), ('complete finite', 'finite sample'),
                         ('min-1 nonmutation', 'min-1 not tested')]:
            self.assertFalse(self.phase(valid.replace(old, new))['passed'])

    def test_failure_and_skip_markers_override_positive_fixture_lines(self):
        valid = phase_line(0) + phase_line(1)
        for trailer in ['FAIL: CUDA exception', 'FAIL check 18', 'SKIP: device unavailable']:
            self.assertFalse(self.phase(valid + trailer)['passed'])

    def test_pure_phase_check_requires_exact_positive_completion(self):
        case = next(c for c in tool.cases('cpu') if c['name'] == 'pure-phase-quotas')
        text = 'PASS 42 pure phase-plan checks; real shared tickets and CUDA remain separate qualification\n'
        self.assertTrue(tool.analyze_component_output(case, text)['passed'])
        for bad in [text.replace('42', '0'), text * 2, 'pure phase-plan checks', text + 'FAIL check 42']:
            self.assertFalse(tool.analyze_component_output(case, bad)['passed'])

    def test_file_read_plan_requires_one_positive_exact_completion(self):
        case = next(c for c in tool.cases('cpu') if c['name'] == 'pure-file-read-plan')
        text = ('PASS 123 pure selected expert read checks; exact source bytes, bounded runs/windows, '
                'scattered slots and pre-I/O rejection; no CUDA or model loaded\n')
        self.assertTrue(tool.analyze_component_output(case, text)['passed'])
        for bad in [text.replace('123', '0'), text * 2, 'pure selected expert read checks',
                    text.replace('exact source bytes', 'sampled source bytes'), text + 'SKIP unavailable']:
            self.assertFalse(tool.analyze_component_output(case, bad)['passed'])

    def test_direct_read_plan_requires_padding_and_both_real_plan_paths(self):
        case = next(c for c in tool.cases('cpu') if c['name'] == 'pure-direct-read-plan')
        text = ('PASS: 123 bounded aligned direct-read planner checks; direct=12 buffered=13, '
                'exact logical bytes, padding/window/extent/guard/invalid-complete-plan proof\n')
        self.assertTrue(tool.analyze_component_output(case, text)['passed'])
        for bad in [text * 2, text.replace('123', '0'), text.replace('direct=12', 'direct=0'),
                    text.replace('buffered=13', 'buffered=0'), text.replace('exact logical', 'sample logical'), text + 'SKIP unavailable']:
            self.assertFalse(tool.analyze_component_output(case, bad)['passed'])

    def test_actual_direct_file_source_needs_completed_os_read(self):
        case = next(c for c in tool.cases('cpu') if c['name'] == 'actual-direct-file-source')
        prefix = ('PASS: 42 actual direct FileSource checks; default-off, padded/exact logical bytes, '
                  'unsupported/unaligned fallback, extent/identity/close proof; ')
        for scope in ['Windows immutable shared source (real truncate not available)',
                      'POSIX actual short read, path replacement, CLOEXEC and descriptor close']:
            text = prefix + scope + '\n'
            self.assertTrue(tool.analyze_component_output(case, text)['passed'])
            for bad in [text.replace('42', '0'), text * 2, text + 'SKIP: direct unsupported', text.replace('identity/close', 'identity')]:
                self.assertFalse(tool.analyze_component_output(case, bad)['passed'])

    def test_direct_optin_scalar_lifecycle_requires_buffered_only_and_complete_cleanup(self):
        case = next(c for c in tool.cases('cuda') if c['name'] == 'scalar-buffered-under-direct-iq1')
        text = ('PASS: GLM scalar top8 selected payload=34603008 exceeds32MiB, poisoned mmap identity, '
                'exact CUDA output, fixed32MiB arena/one-byte refusal, warm reuse, interrupted1MiB ring and cleanup\n'
                'PASS: scalar buffered under direct opt-in completed_direct_reads=0; fixed32MiB/min-1, '
                'poisoned mmap, exact output, warm reuse, interrupted ring and teardown\n')
        self.assertTrue(tool.analyze_component_output(case, text)['passed'])
        for bad in [text.replace('reads=0', 'reads=1'), text.replace('poisoned mmap', 'resident source'),
                    text.replace('ring and teardown', 'ring'), text + 'SKIP: direct unsupported']:
            self.assertFalse(tool.analyze_component_output(case, bad)['passed'])

    def test_direct_optin_short_prefill_preserves_both_full_clamp_lifecycles(self):
        case = next(c for c in tool.cases('cuda') if c['name'] == 'short-buffered-under-direct-iq1')
        clamps = 'PASS: GLM actualdecode1 tails1..8, K8/64expert route union\n' * 2
        text = clamps + ('PASS: short-prefill buffered under direct opt-in completed_direct_reads=0; '
                         'tails1..8, exact CUDA output, min-1/fault/warm/teardown proof\n')
        self.assertTrue(tool.analyze_component_output(case, text)['passed'])
        for bad in [text.replace('reads=0', 'reads=1'), text[len(clamps)//2:], text + 'SKIP unavailable']:
            self.assertFalse(tool.analyze_component_output(case, bad)['passed'])

    def test_direct_wide_phase_requires_complete_modes_and_cold_buffered_narrow_misses(self):
        case = next(c for c in tool.cases('cuda') if c['name'] == 'phase-three-groups-direct')
        extra = ('PASS: wide-only direct phase coexist=0 completed_direct_reads=42 narrow_buffered_reads=12 narrow_direct_reads=0; complete phase exact output/ownership/fault proof\n'
                 'PASS: wide-only direct phase coexist=1 completed_direct_reads=18 narrow_buffered_reads=0 narrow_direct_reads=0; complete phase exact output/ownership/fault proof\n')
        text = phase_line(0) + phase_line(1) + extra
        self.assertTrue(tool.analyze_component_output(case, text)['passed'])
        for bad in [text.replace('reads=42', 'reads=0'), text.replace('narrow_buffered_reads=12', 'narrow_buffered_reads=0'),
                    text.replace('narrow_direct_reads=0', 'narrow_direct_reads=1'), text + extra,
                    phase_line(0) + extra, text + 'SKIP: direct I/O unavailable']:
            self.assertFalse(tool.analyze_component_output(case, bad)['passed'])


if __name__ == '__main__':
    unittest.main()
