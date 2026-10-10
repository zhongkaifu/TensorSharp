import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("qwen38_llama_reference", Path(__file__).resolve().parents[1] / "qwen38-llama-reference.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReferenceEvidence(unittest.TestCase):
    def test_prompt_ids_are_preserved_with_no_cache_or_repetition_adjustment(self):
        record = {"synthetic": False, "prompt_tokens": [1, 42, 248320]}
        request = module.request_body(record, 100)
        self.assertEqual(record["prompt_tokens"], request["prompt"])
        self.assertFalse(request["cache_prompt"])
        self.assertEqual(1.0, request["repeat_penalty"])
        self.assertEqual(0.0, request["temperature"])

    def test_missing_or_synthetic_conditioning_cannot_become_a_reference(self):
        for record in ({}, {"synthetic": True, "prompt_tokens": [1]}, {"synthetic": False, "prompt_tokens": [True]}):
            with self.assertRaises(ValueError): module.request_body(record, 100)

    def test_exact_math_and_eos_are_required_for_quality(self):
        result = {"tokens": [19, 17, 248046], "tokens_predicted": 3, "content": "\n42", "stop_type": "eos"}
        self.assertTrue(module.classify(result, 32, "42")["quality_passed"])
        result["stop_type"] = "limit"
        check = module.classify(result, 32, "42")
        self.assertTrue(check["token_evidence_valid"])
        self.assertFalse(check["quality_passed"])

    def test_long_free_output_is_never_an_automatic_semantic_pass(self):
        result = {"tokens": [1] * 100, "tokens_predicted": 100, "content": "words", "stop_type": "limit"}
        check = module.classify(result, 100)
        self.assertTrue(check["token_evidence_valid"])
        self.assertFalse(check["answer_complete"])
        self.assertIsNone(check["quality_passed"])
        result["tokens_predicted"] = 99
        self.assertFalse(module.classify(result, 100)["token_evidence_valid"])


if __name__ == "__main__": unittest.main()
