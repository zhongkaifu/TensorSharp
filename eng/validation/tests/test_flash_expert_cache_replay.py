import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    'replay', Path(__file__).resolve().parents[1] / 'flash-expert-cache-replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class RouteReplayTests(unittest.TestCase):
    def test_duplicate_miss_is_one_upload_and_then_a_hit(self):
        for decay in (0, 32):
            cache = replay.Cache(2, 4, decay)
            self.assertEqual(cache.access([0, 0]), (1, 1))
            self.assertEqual(cache.access([0, 0]), (2, 0))

    def test_later_row_hit_is_protected_before_earlier_miss(self):
        for decay in (0, 32):
            cache = replay.Cache(2, 4, decay)
            cache.access([0, 1])
            self.assertEqual(cache.access([2, 0]), (1, 1))
            self.assertEqual(set(cache.slots), {0, 2})

    def test_counts_saturate_and_age_like_native_uint16(self):
        cache = replay.Cache(2, 4, 32)
        cache.counts[0] = 65535
        cache.access([0])
        self.assertEqual(cache.counts[0], 65535)
        for _ in range(31):
            cache.access([1])
        self.assertEqual(cache.counts[0], 32767)
        self.assertEqual(cache.counts[1], 16)

    def test_recreation_drops_old_residency(self):
        total, rows = replay.replay([
            '[HOSTMOE-ROUTE-CREATE] layer=0 slots=2 experts=4',
            '[HOSTMOE-ROUTE] layer=0 ids=0,1',
            '[HOSTMOE-ROUTE] layer=0 ids=1,0',
            '[HOSTMOE-ROUTE-CREATE] layer=0 slots=2 experts=4',
            '[HOSTMOE-ROUTE] layer=0 ids=0,1',
        ], 0)
        self.assertEqual(rows, [(0, 2), (2, 0), (0, 2)])
        self.assertEqual((total['hits'], total['misses'], total['creations']), (2, 4, 2))

    def test_incomplete_and_invalid_trace_is_rejected(self):
        for lines in (
            [],
            ['[HOSTMOE-ROUTE] layer=0 ids=0'],
            ['[HOSTMOE-ROUTE-CREATE] layer=0 slots=2 experts=4', '[HOSTMOE-ROUTE] layer=0 ids=0,'],
            ['[HOSTMOE-ROUTE-CREATE] layer=0 slots=2 experts=4', '[HOSTMOE-ROUTE] layer=0 ids=4'],
            ['[HOSTMOE-ROUTE-CREATE] layer=0 slots=2 experts=4', '[HOSTMOE-ROUTE] layer=0 ids=0,1,2'],
        ):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                replay.replay(lines, 0)


if __name__ == '__main__':
    unittest.main()
