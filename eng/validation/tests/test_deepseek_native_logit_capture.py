import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('capture',Path(__file__).resolve().parents[1]/'deepseek-native-logit-capture.py')
capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)


class MappingRangeTests(unittest.TestCase):
    def test_only_exact_checkpoint_paths_are_advised(self):
        with tempfile.TemporaryDirectory() as directory:
            path=str((Path(directory)/'model.gguf').resolve())
            maps=f'00001000-00003000 r--p 00000000 00:00 1 {path}\n00004000-00005000 rw-p 00000000 00:00 2 /unrelated'
            self.assertEqual(capture.checkpoint_mapping_ranges(maps,[path],4096),[dict(address=4096,length=8192,path=path)])

    def test_writable_or_unaligned_model_ranges_are_refused(self):
        path=str(Path('model.gguf').resolve())
        for addresses,permissions in [('00001000-00003000','rw-p'),('00001001-00003000','r--p'),('00003000-00001000','r--p')]:
            with self.subTest(addresses=addresses,permissions=permissions),self.assertRaises(ValueError):
                capture.checkpoint_mapping_ranges(f'{addresses} {permissions} 0 00:00 1 {path}',[path],4096)

    def test_missing_mapping_is_not_counted_as_eviction(self):
        with self.assertRaises(ValueError):
            capture.checkpoint_mapping_ranges('00001000-00003000 r--p 0 00:00 1 /another', [Path('model.gguf')],4096)


if __name__=='__main__': unittest.main()
