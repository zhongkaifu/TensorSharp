import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location('activity_report', pathlib.Path(__file__).parents[1]/'cuda-activity-report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ActivityReportTests(unittest.TestCase):
    def rows(self):
        return [dict(kind='mark', timestamp=100, label='decode/1/2/begin'),
                dict(kind='mark', timestamp=200, label='decode/1/2/end'),
                dict(kind='kernel', start=110, end=160, device=0, name='matvec'),
                dict(kind='api', start=105, end=165, name='cudaGraphLaunch'),
                dict(kind='kernel', start=30, end=60, device=0, name='prefill'),
                dict(kind='summary', records=3, dropped=0, errors=0)]

    def test_unordered_callback_records_use_timestamp_windows_and_keep_overlap_separate(self):
        data = report.summarize(list(reversed(self.rows())))
        self.assertEqual(1, data['steps'])
        self.assertEqual(['matvec'], [x['name'] for x in data['kernel']])
        self.assertEqual(0.0001, data['mean_marked_wall_ms'])
        self.assertEqual(0.00005, data['kernel'][0]['ms_per_step'])
        self.assertEqual(0.00006, data['api'][0]['ms_per_step'])

    def test_dropped_records_cannot_be_a_valid_profile(self):
        rows=self.rows();rows[-1]['dropped']=1
        with self.assertRaisesRegex(ValueError, 'dropped'):report.summarize(rows)

    def test_incomplete_or_crossing_window_is_rejected(self):
        for rows in [self.rows()[1:],self.rows()]:
            if len(rows)==6:rows[2]['end']=210
            with self.assertRaises(ValueError):report.summarize(rows)

    def test_marker_without_any_kernel_is_not_a_pass(self):
        with self.assertRaisesRegex(ValueError, 'no kernels'):
            report.summarize([r for r in self.rows() if r['kind']!='kernel'])


if __name__ == '__main__':unittest.main()
