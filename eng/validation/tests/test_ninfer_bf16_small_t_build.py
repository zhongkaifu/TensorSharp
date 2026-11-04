import copy
import importlib.util
import json
import pathlib
import shutil
import tempfile
import unittest


PATH = pathlib.Path(__file__).resolve().parents[1] / "build-ninfer-bf16-small-t.py"
SPEC = importlib.util.spec_from_file_location("ninfer_bf16_small_t_builder", PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
UPSTREAM = PATH.parents[2] / "TensorSharp.Backends.Cuda/native/ninfer/bf16_small_t/upstream"


class Bf16SmallTBuildContracts(unittest.TestCase):
    def test_original_files_match_git_objects(self):
        actual = BUILDER.verify_upstream(UPSTREAM)
        self.assertEqual(9, actual["filesVerified"])
        self.assertEqual([], actual["modifiedUpstreamFiles"])

    def test_changed_original_header_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = pathlib.Path(directory) / "upstream"
            shutil.copytree(UPSTREAM, folder)
            path = folder / "ops/linear/bf16/bf16_small_t.cuh"
            path.write_bytes(path.read_bytes() + b"\n// unexpected modification\n")
            with self.assertRaisesRegex(ValueError, "Pristine Ignis"):
                BUILDER.verify_upstream(folder)

    def test_changed_manifest_cannot_repin_original(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = pathlib.Path(directory) / "upstream"
            shutil.copytree(UPSTREAM, folder)
            manifest = json.loads((folder / "manifest.json").read_text())
            manifest["revision"] = "0" * 40
            (folder / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                BUILDER.verify_upstream(folder)

    def test_original_policy_is_an_aligned_struct_not_flat_pointer(self):
        args = BUILDER.parse_parameters(".param .u64 x, .param .u64 w, .param .align 8 .b8 o[16]")
        self.assertEqual([0, 8, 16], [x["offset"] for x in args])
        self.assertEqual([8, 8, 16], [x["size"] for x in args])
        self.assertEqual("b8", args[2]["type"])

    def test_driver_pointer_qualifiers_are_not_new_parameters(self):
        args = BUILDER.parse_parameters(".param .u64 .ptr .global .align 1 x, .param .align 8 .b8 o[8]")
        self.assertEqual([0, 8], [x["offset"] for x in args])
        self.assertEqual([8, 8], [x["size"] for x in args])

    def test_scalar_and_zero_length_payloads_are_refused(self):
        for item in (".param .u32 x", ".param .b8 x[0]", ".param .align 3 .b8 x[8]",
                     ".param .u64 x[2]", ".param .u64 x trailing"):
            with self.subTest(item=item), self.assertRaises(ValueError):
                BUILDER.parse_parameters(item)

    def test_missing_duplicate_unknown_geometry_and_bad_abis_are_refused(self):
        self.assertEqual(7, BUILDER.self_test()["rejectedMalformedAbiCases"])


if __name__ == "__main__":
    unittest.main()
