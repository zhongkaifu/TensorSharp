# Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('pipeline_timeline', Path(__file__).resolve().parents[1] / 'analyze-moe-pipeline-timeline.py')
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def fixture():
    return '\n'.join([
        tool.HEADER + 'seq=33 chunks=4 pinned=4096 total_ms=0.001 gate_stream_start_ns=100 gate_stream_end_ns=300 down_read_overlap_sum_ms=0.000 down_copy_overlap_sum_ms=0.000',
        tool.SPAN + 'seq=33 chunk=0 part=0 bytes=16 read_start_ns=10 read_end_ns=20 copy_start_ns=30 copy_end_ns=40',
        tool.SPAN + 'seq=33 chunk=1 part=1 bytes=16 read_start_ns=20 read_end_ns=40 copy_start_ns=50 copy_end_ns=60',
        tool.SPAN + 'seq=33 chunk=2 part=2 bytes=16 read_start_ns=80 read_end_ns=170 copy_start_ns=180 copy_end_ns=240',
        tool.SPAN + 'seq=33 chunk=3 part=2 bytes=16 read_start_ns=120 read_end_ns=200 copy_start_ns=250 copy_end_ns=340'])


class TimelineTests(unittest.TestCase):
    def test_observed_union_does_not_double_count_reads(self):
        row = tool.analyze(fixture())['calls'][0]
        self.assertEqual(row['down_read_overlap_union_ns'], 100)
        self.assertEqual(row['down_read_overlap_sum_ns'], 150)
        self.assertEqual(row['down_copy_overlap_union_ns'], 110)
        self.assertFalse(row['parallel_read_geometry_observed'])

    def test_parallel_read_geometry_requires_more_jobs_than_transfers(self):
        text = fixture().replace('chunks=4 ', 'chunks=4 read_jobs=8 max_read_bytes=1024 ')
        result = tool.analyze(text)
        self.assertTrue(result['parallel_read_geometry_observed'])
        self.assertEqual(result['calls'][0]['read_jobs'], 8)
        self.assertFalse(tool.analyze(text.replace('read_jobs=8', 'read_jobs=4'))['parallel_read_geometry_observed'])

    def test_incomplete_or_invalid_parallel_read_geometry_rejected(self):
        for fields in ('read_jobs=8', 'max_read_bytes=1024',
                       'read_jobs=3 max_read_bytes=1024',
                       'read_jobs=8 max_read_bytes=0',
                       'read_jobs=8 max_read_bytes=16777217'):
            with self.assertRaises(ValueError):
                tool.analyze(fixture().replace('chunks=4 ', 'chunks=4 ' + fields + ' '))

    def test_missing_and_empty_rejected(self):
        for text in ('', fixture().rsplit('\n', 1)[0]):
            with self.assertRaises(ValueError): tool.analyze(text)

    def test_wrong_order_rejected(self):
        with self.assertRaises(ValueError): tool.analyze(fixture().replace('read_end_ns=170', 'read_end_ns=190'))

    def test_bad_bounds_or_nonfinite_rejected(self):
        for old, new in [('pinned=4096', 'pinned=16777217'), ('total_ms=0.001', 'total_ms=nan'), ('gate_stream_end_ns=300', 'gate_stream_end_ns=0')]:
            with self.assertRaises(ValueError): tool.analyze(fixture().replace(old, new))

    def test_false_reported_overlap_rejected(self):
        with self.assertRaises(ValueError): tool.analyze(fixture().replace('down_copy_overlap_sum_ms=0.000', 'down_copy_overlap_sum_ms=3.000'))

    def test_mismatched_or_duplicate_chunk_rejected(self):
        for text in (fixture().replace('seq=33 chunk=2', 'seq=32 chunk=2'), fixture().replace('chunk=3', 'chunk=2')):
            with self.assertRaises(ValueError): tool.analyze(text)


if __name__ == '__main__': unittest.main()
