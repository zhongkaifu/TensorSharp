import importlib.util
import json
import pathlib
import shutil
import tempfile
import unittest


PATH = pathlib.Path(__file__).resolve().parents[1] / "build-ninfer-bf16-mma.py"
SPEC = importlib.util.spec_from_file_location("ninfer_bf16_mma_builder", PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
UPSTREAM = PATH.parents[2] / "TensorSharp.Backends.Cuda/native/ninfer/bf16_mma/upstream"


class Bf16MmaBuildContracts(unittest.TestCase):
    def test_original_files_match_git_objects(self):
        actual = BUILDER.verify_upstream(UPSTREAM)
        self.assertEqual(5, actual["filesVerified"])
        self.assertEqual([], actual["modifiedUpstreamFiles"])

    def test_changed_original_core_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = pathlib.Path(directory) / "upstream"
            shutil.copytree(UPSTREAM, folder)
            core = folder / "ops/linear/bf16/bf16_gemm_mma.cuh"
            core.write_bytes(core.read_bytes() + b"\n// modification\n")
            with self.assertRaisesRegex(ValueError, "Pristine Ignis"):
                BUILDER.verify_upstream(folder)

    def test_manifest_cannot_repin_modified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = pathlib.Path(directory) / "upstream"
            shutil.copytree(UPSTREAM, folder)
            manifest = json.loads((folder / "manifest.json").read_text())
            manifest["revision"] = "0" * 40
            (folder / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                BUILDER.verify_upstream(folder)

    def test_raw_f32_policy_and_runtime_tokens_abi(self):
        args = BUILDER.parse_parameters(
            ".param .u64 x, .param .u64 w, .param .align 8 .b8 out[8], .param .u32 tokens")
        self.assertEqual([0, 8, 16, 24], [x["offset"] for x in args])
        self.assertEqual([8, 8, 8, 4], [x["size"] for x in args])

    def test_original_bf16_policy_moves_tokens_to_offset_32(self):
        args = BUILDER.parse_parameters(
            ".param .u64 x, .param .u64 w, .param .align 8 .b8 out[16], .param .u32 tokens")
        self.assertEqual([0, 8, 16, 32], [x["offset"] for x in args])
        self.assertEqual([8, 8, 16, 4], [x["size"] for x in args])

    def test_pointer_qualifiers_do_not_change_storage(self):
        args = BUILDER.parse_parameters(
            ".param .u64 .ptr .global .align 1 x, .param .u32 tokens")
        self.assertEqual([0, 8], [x["offset"] for x in args])
        for value in (".param .b8 x[0]", ".param .align 3 .b8 x[8]",
                      ".param .u64 x[2]", ".param .u64 x trailing"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BUILDER.parse_parameters(value)

    def test_four_entries_and_wrong_tokens_policy_or_geometry_refused(self):
        result = BUILDER.self_test()
        self.assertTrue(result["validFourEntryAbi"])
        self.assertEqual(9, result["rejectedMalformedAbiCases"])
        self.assertEqual("NOT_RUN", result["GPU"])


if __name__ == "__main__":
    unittest.main()
