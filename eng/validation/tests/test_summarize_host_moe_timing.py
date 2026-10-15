import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location('host_moe_timing_summary',
    Path(__file__).parents[1] / 'summarize-host-moe-timing.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def line(**changes):
    fields = {key: '0' for key in MODULE.INTEGER_KEYS + MODULE.MILLISECOND_KEYS}
    fields.update(layer='3', rows='85', bulk='1', outcome='completed', scope=MODULE.SCOPE,
                  total_ms='50.125', file_pipeline_ms='40.0', upload_enqueue_ms='5.0',
                  compute_ms='8.0', file_bytes='12345', graph_submissions='1')
    fields.update(changes)
    return MODULE.MARKER + ' ' + ' '.join(f'{key}={value}' for key, value in fields.items())


class TimingSummaryTests(unittest.TestCase):
    def test_separates_bulk_and_scalar_without_claiming_decode(self):
        result = MODULE.summarize('noise\n' + line() + '\n' + line(layer='4') + '\n' + line(rows='1', bulk='0'))
        groups = {r['ExecutionKind']: r for r in result['Groups']}
        self.assertEqual(groups['genuine-wide-prefill']['Callbacks'], 2)
        self.assertEqual(groups['genuine-wide-prefill']['Rows'], 170)
        self.assertEqual(groups['genuine-wide-prefill']['Counters']['file_bytes'], 24690)
        self.assertEqual(groups['scalar-phase-unresolved']['Callbacks'], 1)
        self.assertEqual(len(result['OrdinalShapeBlocks']), 2)

    def test_keeps_nested_intervals_not_subtracting_or_clamping(self):
        result = MODULE.summarize(line(plan_ms='49.0', allocation_ms='48.0', wide_replace_ms='47.0'))
        values = result['Groups'][0]['SumHostIntervalsMs']
        self.assertEqual(values['total_ms'], 50.125)
        self.assertGreater(sum(values.values()), values['total_ms'])

    def test_joint_and_narrow_are_explicit(self):
        result = MODULE.summarize(line(rows='3', joint='1', bulk='0') + '\n' + line(rows='4', bulk='0'))
        self.assertEqual({r['ExecutionKind'] for r in result['Groups']}, {'joint-decode', 'nonjoint-narrow-prefill'})

    def test_failures_remain_observed_failures(self):
        result = MODULE.summarize(line(outcome='exception'))
        self.assertFalse(result['AllCallbacksCompleted'])
        self.assertEqual(result['Groups'][0]['Outcomes']['exception'], 1)

    def test_absent_or_partial_coverage_rejected(self):
        for value in ('unrelated', MODULE.MARKER + ' rows=85', line().replace('total_ms=50.125 ', '')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE.summarize(value)

    def test_duplicate_or_unknown_fields_rejected(self):
        for value in (line() + ' rows=2', line() + ' kernel_ms=5', line(scope='cuda_kernel')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE.summarize(value)

    def test_invalid_numbers_or_shapes_rejected(self):
        for changes in ({'total_ms': 'nan'}, {'compute_ms': '-1'}, {'rows': '0'},
                        {'bulk': '2'}, {'joint': '1'}, {'file_bytes': '-1'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                MODULE.summarize(line(**changes))


if __name__ == '__main__':
    unittest.main()
