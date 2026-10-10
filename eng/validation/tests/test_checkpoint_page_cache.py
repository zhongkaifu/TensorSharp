import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('cache',Path(__file__).resolve().parents[1]/'checkpoint-page-cache.py')
cache=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


@unittest.skipUnless(sys.platform=='linux','Actual mincore and fadvise require Linux')
class CacheTests(unittest.TestCase):
    def test_eviction_observes_residency_without_modifying_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.gguf'
            data=bytes(range(256))*321+17*b'x'
            path.write_bytes(data)
            for evict in (False,True):
                report=cache.cache_state([path],evict)
                self.assertEqual(path.read_bytes(),data)
                self.assertEqual(report['files'][0]['bytes'],len(data))
                pages=report['files'][0]['pages']
                self.assertEqual(pages,(len(data)+report['page_bytes']-1)//report['page_bytes'])
                for key in ('resident_before','resident_after'):
                    self.assertGreaterEqual(report['files'][0][key],0)
                    self.assertLessEqual(report['files'][0][key],pages)


if __name__=='__main__': unittest.main()
