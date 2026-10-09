import importlib.util
from pathlib import Path
import tempfile
import unittest

available = importlib.util.find_spec("gguf") is not None
if available:
    import numpy as np
    from gguf import GGUFReader, GGUFWriter
    spec = importlib.util.spec_from_file_location("gguf_tensor_variant",
        Path(__file__).resolve().parents[1] / "create-gguf-tensor-variant.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@unittest.skipUnless(available, "The unmodified gguf Python package is required")
class TensorVariantTests(unittest.TestCase):
    def fixture(self, path, donor=False, mutation=None):
        writer = GGUFWriter(path, "llama")
        writer.add_uint32("llama.embedding_length", 2)
        if mutation == "split": writer.add_uint16("split.count", 2)
        if mutation == "alignment": writer.add_custom_alignment(64)
        writer.add_array("tokenizer.ggml.tokens", ["a", "changed" if mutation == "tokenizer" else "b"])
        writer.add_tensor("token_embd.weight", np.full((3 if mutation == "shape" else 2, 2), 2 if donor else 1, dtype=np.float16))
        writer.add_tensor("output_norm.weight", np.full(2, 3 if mutation == "norm" else 1, dtype=np.float32))
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()

    def test_exact_selected_substitution_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, donor = root / "a.gguf", root / "b.gguf"
            self.fixture(source); self.fixture(donor, True)
            a, b = module.sha_file(source), module.sha_file(donor)
            report = module.create(source, donor, a, b, ["token_embd.weight"], root / "out")
            self.assertTrue(report["complete"])
            self.assertFalse(report["original_model_fixed"])
            self.assertEqual(["token_embd.weight"], [t["name"] for t in report["tensors"] if t["donor"]])
            self.assertEqual((a, b), (module.sha_file(source), module.sha_file(donor)))
            derived = GGUFReader(report["output"])
            self.assertIn("DIAGNOSTIC", derived.fields["general.name"].contents())
            self.assertEqual(2, float(derived.tensors[0].data.flat[0]))
            self.assertEqual(1, float(derived.tensors[1].data.flat[0]))
            del derived  # Release the mapped file before Windows temp cleanup.

    def test_refuses_incompatible_or_unpinned_donor_before_output(self):
        for mutation in ("tokenizer", "shape", "norm", "hash", "missing", "duplicate", "split"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source, donor = root / "a.gguf", root / "b.gguf"
                self.fixture(source); self.fixture(donor, True, mutation)
                selected = ["missing"] if mutation == "missing" else ["token_embd.weight"]
                if mutation == "duplicate": selected *= 2
                with self.assertRaises(ValueError):
                    module.create(source, donor, module.sha_file(source),
                        "0" * 64 if mutation == "hash" else module.sha_file(donor), selected, root / "out")
                self.assertFalse((root / "out").exists())

    def test_custom_alignment_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, donor = root / "a.gguf", root / "b.gguf"
            self.fixture(source, mutation="alignment"); self.fixture(donor, True)
            report = module.create(source, donor, module.sha_file(source), module.sha_file(donor),
                ["token_embd.weight"], root / "out")
            derived = GGUFReader(report["output"])
            self.assertEqual(64, derived.alignment)
            del derived


if __name__ == "__main__":
    unittest.main()
