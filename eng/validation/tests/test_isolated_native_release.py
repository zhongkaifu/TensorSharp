import importlib.util
from pathlib import Path
import tarfile
import unittest

path=Path(__file__).resolve().parents[1]/'build-isolated-native-release.py'
spec=importlib.util.spec_from_file_location('isolated_native_release',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class IsolatedNativeReleaseTests(unittest.TestCase):
    def test_accepts_separate_new_workspace_tree(self):
        target=Path('/workspace/tensorsharp-v6-native-unexecuted-unit-fixture')
        self.assertEqual(target.resolve(),module.safe_work_root(target,Path('/workspace/tensorsharp-no-patch-20260915')))

    def test_refuses_frozen_tree_and_descendant(self):
        reference=Path('/workspace/tensorsharp-no-patch-20260915')
        for target in [reference,reference/'v6',reference.parent]:
            with self.assertRaises(ValueError):module.safe_work_root(target,reference)

    def test_archive_rejects_escape_links_duplicates_and_unowned_members(self):
        good=tarfile.TarInfo('TensorSharp.GGML.Native/owned.cpp')
        module.validate_members([good],{good.name:{}})
        for name in ['../owned.cpp','/owned.cpp','ExternalProjects/ggml/src/ggml.c']:
            bad=tarfile.TarInfo(name)
            with self.assertRaises(ValueError):module.validate_members([bad],{name:{}})
        link=tarfile.TarInfo(good.name);link.type=tarfile.SYMTYPE;link.linkname='/tmp/foreign'
        with self.assertRaises(ValueError):module.validate_members([link],{good.name:{}})
        with self.assertRaises(ValueError):module.validate_members([good,good],{good.name:{}})

    def test_later_success_cannot_hide_original_tp_failure(self):
        steps=[{'name':'tp7-checkpoint-shape','status':'failed'}, {'name':'retention','status':'passed'}]
        self.assertEqual('failed',module.result_status(steps,[{'passed':True}]))

    def test_hardware_skip_is_not_a_pass(self):
        self.assertEqual('incomplete_native_gates',module.result_status([{'status':'unavailable'}],[{'passed':True}]))

    def test_failed_provenance_blocks_success(self):
        self.assertEqual('failed',module.result_status([{'status':'passed'}],[{'passed':False}]))
        self.assertEqual('native_gates_passed_full_model_not_run',module.result_status([{'status':'passed'}],[{'passed':True}]))


if __name__=='__main__':unittest.main()
