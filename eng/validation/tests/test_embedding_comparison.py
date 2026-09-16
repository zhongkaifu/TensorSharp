import copy
import importlib.util
import math
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "validate-release-embeddings.py"
SPEC = importlib.util.spec_from_file_location("release_embeddings", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EmbeddingComparisonTests(unittest.TestCase):
    def setUp(self):
        self.reference = {"dimensions": 384, "usage": {"prompt_tokens": 13, "total_tokens": 13},
                          "rankings": {"retrieval": True}, "vectors": [[1.0] + [0.0] * 383]}

    def test_identical_outputs_pass_both_scopes(self):
        for scope in ("same-backend", "cross-backend"):
            result = MODULE.compare_reference(self.reference, self.reference, scope)
            self.assertEqual("passed", result["status"])

    def test_documented_cross_backend_contract_does_not_erase_strict_failure(self):
        candidate = copy.deepcopy(self.reference)
        tail = 0.001
        candidate["vectors"][0] = [math.sqrt(1 - 383 * tail * tail)] + [tail] * 383
        self.assertEqual("failed", MODULE.compare_reference(candidate, self.reference, "same-backend")["status"])
        result = MODULE.compare_reference(candidate, self.reference, "cross-backend")
        self.assertEqual("passed", result["status"])
        self.assertEqual(.999, result["minimum_cosine"])
        self.assertEqual(.005, result["maximum_component_error"])

    def test_component_error_fails_even_when_cosine_passes(self):
        candidate = copy.deepcopy(self.reference)
        candidate["vectors"][0][:2] = [math.sqrt(1 - .006**2), .006]
        result = MODULE.compare_reference(candidate, self.reference, "cross-backend")
        self.assertGreater(result["min_cosine"], .999)
        self.assertEqual("failed", result["status"])
        self.assertTrue(any("component error" in reason for reason in result["failures"]))

    def test_accounting_and_rankings_are_independent_gates(self):
        for field, value in (("usage", {"prompt_tokens": 12, "total_tokens": 12}),
                             ("rankings", {"retrieval": False}), ("dimensions", 383)):
            candidate = copy.deepcopy(self.reference)
            candidate[field] = value
            self.assertEqual("failed", MODULE.compare_reference(candidate, self.reference, "cross-backend")["status"])

    def test_truncated_and_nonfinite_vectors_fail(self):
        for vector in ([1.0], [float("nan")] + [0.0]*383):
            candidate = copy.deepcopy(self.reference)
            candidate["vectors"] = [vector]
            self.assertEqual("failed", MODULE.compare_reference(candidate, self.reference, "cross-backend")["status"])


if __name__ == "__main__":
    unittest.main()
