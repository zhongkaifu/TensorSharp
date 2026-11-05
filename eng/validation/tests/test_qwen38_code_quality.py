import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("prime_quality", Path(__file__).parents[1] / "qwen38-code-quality.py")
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)

CORRECT = '''
def is_prime(n: int) -> bool:
    """Trial division with exact integer bounds."""
    if n < 2:
        return False
    for d in range(2, math.isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

def primes_below(n: int) -> list[int]:
    if n <= 2:
        return []
    sieve = [True] * n
    sieve[0:2] = [False, False]
    p = 2
    while p * p < n:
        if sieve[p]:
            sieve[p*p:n:p] = [False] * len(range(p*p, n, p))
        p += 1
    return [i for i, prime in enumerate(sieve) if prime]
'''


class CodeQualityTests(unittest.TestCase):
    def test_correct_sieve_checks_negative_prime_square_and_exclusive_boundaries(self):
        # Avoid installing worker OS limits in the test runner itself.
        functions, identities, issues = quality.extract_functions(CORRECT)
        quality.validate_functions(functions)
        self.assertEqual([], issues)
        checks = quality.check_functions(functions)
        self.assertEqual({"PASS"}, {row["status"] for row in checks.values()})
        self.assertGreater(checks["is_prime"]["passed_cases"], 280)
        self.assertGreater(checks["primes_below"]["passed_cases"], 150)
        self.assertEqual({"is_prime", "primes_below"}, set(identities))

    def test_off_by_one_and_square_bound_bugs_have_real_counterexamples(self):
        for source, name, argument in [
            (CORRECT.replace("range(2, math.isqrt(n) + 1)", "range(2, math.isqrt(n))"), "is_prime", 4),
            (CORRECT.replace("return [i for i, prime in enumerate(sieve) if prime]",
                             "return [i for i, prime in enumerate(sieve) if prime and i < n-1]"), "primes_below", 3),
            (CORRECT.replace("if n < 2:\n        return False", "if n < 2:\n        return True"), "is_prime", -17),
        ]:
            functions, _, _ = quality.extract_functions(source)
            quality.validate_functions(functions)
            result = quality.check_functions(functions)[name]
            self.assertEqual("FAIL", result["status"])
            self.assertEqual(argument, result["counterexample"]["input"])

    def test_append_loop_and_annotation_stripping_are_supported(self):
        source = CORRECT.replace("-> list[int]", "-> List[int]").replace(
            "return [i for i, prime in enumerate(sieve) if prime]",
            "result: List[int] = []\n    for i in range(2, n):\n        if sieve[i]:\n            result.append(i)\n    return result")
        functions, _, _ = quality.extract_functions(source)
        quality.validate_functions(functions)
        self.assertEqual([2, 3, 5, 7], quality.Interpreter(functions).call("primes_below", [10]))

    def test_complete_supported_functions_pass_and_decorators_are_rejected(self):
        source = CORRECT.replace("math.isqrt(n)", "int(n ** 0.5)")
        result = quality.diagnose_text(source, quality.sha(source.encode()), "stop")
        self.assertEqual("PASS", result["status"])
        self.assertIn("annotation_stripped_ast_sha256", result["functions"]["is_prime"])
        source = "@open('/tmp/must-not-execute', 'w')\n" + source.lstrip()
        result = quality.diagnose_text(source, quality.sha(source.encode()), "stop")
        self.assertEqual("UNVERIFIED", result["status"])
        self.assertIn("decorator", result["reason"])

    def test_malicious_import_open_getattr_and_dunder_are_rejected(self):
        payloads = ["import os\n    return True", "return open('/tmp/never-created', 'w')",
                    "return getattr(n, '__class__')", "return n.__class__",
                    "return __import__('os')", "return eval('1')"]
        for payload in payloads:
            source = "def is_prime(n):\n    " + payload + "\n\ndef primes_below(n):\n    return []\n"
            with self.subTest(payload=payload):
                functions, _, _ = quality.extract_functions(source)
                with self.assertRaises(quality.Unsupported):
                    quality.validate_functions(functions)

    def test_no_module_level_code_is_executed(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "must-not-exist"
            source = "import os\nopen(" + repr(str(marker)) + ", 'w').write('bad')\n" + CORRECT
            result = quality.diagnose_text(source, quality.sha(source.encode()), "stop")
            self.assertFalse(marker.exists())
            self.assertEqual("UNVERIFIED", result["status"])
            self.assertTrue(any("Import" in reason for reason in result["source_issues"]))
            self.assertEqual("PASS", result["function_checks"]["is_prime"]["status"])

    def test_truncated_reply_stays_unverified_even_with_complete_functions(self):
        source = "```python\n" + CORRECT + "\nif __name__ == '__main__':\n    x = ("
        result = quality.diagnose_text(source, quality.sha(source.encode()), "length")
        self.assertEqual("UNVERIFIED", result["status"])
        self.assertEqual("PASS", result["function_checks"]["primes_below"]["status"])
        self.assertIn("response_not_finished_with_stop", result["source_issues"])
        self.assertIn("unclosed_code_fence", result["source_issues"])

    def test_truncated_function_and_hash_mismatch_are_not_passes(self):
        source = "def is_prime(n):\n    return True\n\ndef primes_below(n):\n    return [x for"
        result = quality.diagnose_text(source, quality.sha(source.encode()), "length")
        self.assertEqual("UNVERIFIED", result["status"])
        self.assertIn("not_closed", result["reason"])
        result = quality.diagnose_text(CORRECT, "0" * 64, "stop")
        self.assertEqual("stored_content_hash_mismatch", result["reason"])

    def test_infinite_loop_and_allocation_bomb_are_bounded(self):
        source = CORRECT.replace("if n < 2:", "while True:\n        pass\n    if n < 2:", 1)
        result = quality.diagnose_text(source, quality.sha(source.encode()), "stop", timeout=.05)
        self.assertEqual("UNVERIFIED", result["status"])
        self.assertIn(result.get("reason"), ("worker_timeout", "worker_resource_limit_or_error"))
        bomb = CORRECT.replace("if n < 2:", "x = [0] * (10 ** 12)\n    if n < 2:", 1)
        functions, _, _ = quality.extract_functions(bomb)
        quality.validate_functions(functions)
        with self.assertRaisesRegex(quality.BudgetExceeded, "container_size"):
            quality.Interpreter(functions).call("is_prime", [3])
        many = CORRECT.replace("if n < 2:", "for _ in range(1000):\n        x = [0] * 50000\n    if n < 2:", 1)
        functions, _, _ = quality.extract_functions(many)
        quality.validate_functions(functions)
        with self.assertRaisesRegex(quality.BudgetExceeded, "total_allocation"):
            quality.Interpreter(functions).call("is_prime", [3])

    def test_reserved_names_and_unbounded_recursion_do_not_escape(self):
        shadowed = CORRECT.replace("if n < 2:", "range = 3\n    if n < 2:", 1)
        functions, _, _ = quality.extract_functions(shadowed)
        with self.assertRaises(quality.Unsupported): quality.validate_functions(functions)
        recursive = CORRECT.replace("if n < 2:", "return is_prime(n)\n    if n < 2:", 1)
        functions, _, _ = quality.extract_functions(recursive)
        quality.validate_functions(functions)
        with self.assertRaisesRegex(quality.BudgetExceeded, "depth"):
            quality.Interpreter(functions).call("is_prime", [3])

    def test_unimplemented_in_place_alias_semantics_are_unverified_not_false_failures(self):
        source = CORRECT.replace("if n < 2:", "a = [0]\n    b = a\n    a += [1]\n    if n < 2:", 1)
        result = quality.diagnose_text(source, quality.sha(source.encode()), "stop")
        self.assertEqual("UNVERIFIED", result["status"])
        self.assertEqual("UNVERIFIED", result["function_checks"]["is_prime"]["status"])

    def test_enumerate_is_lazy_so_mutating_its_source_cannot_false_pass(self):
        source = CORRECT.replace("if n < 2:", "a = [0]\n    for i, value in enumerate(a):\n        a.append(0)\n    if n < 2:", 1)
        functions, _, _ = quality.extract_functions(source)
        quality.validate_functions(functions)
        with self.assertRaises(quality.BudgetExceeded):
            quality.Interpreter(functions).call("is_prime", [3])
        nested = CORRECT.replace("if n < 2:", "a = enumerate(enumerate(range(50000)))\n    if n < 2:", 1)
        functions, _, _ = quality.extract_functions(nested)
        quality.validate_functions(functions)
        with self.assertRaisesRegex(quality.Unsupported, "nested_enumerate"):
            quality.Interpreter(functions).call("is_prime", [3])

    def test_supplement_preserves_failed_original_verdict_and_raw_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw.jsonl"
            raw.write_text(json.dumps({"kind": "case", "scenario": "long_generation", "repeat": 0,
                "pass": False, "verdict": "FAIL", "metrics": {"per_prompt": [{"kind": "code"}]},
                "requests": [{"content": CORRECT, "content_sha256": quality.sha(CORRECT.encode()),
                              "finish_reason": "stop", "request_sha256": "request-pin"}]}) + "\n")
            before = raw.read_bytes()
            result = quality.analyze([raw])
            self.assertEqual(before, raw.read_bytes())
            row = result["records"][0]
            self.assertFalse(row["original_pass"])
            self.assertEqual("FAIL", row["original_verdict"])
            self.assertEqual("request-pin", row["request_sha256"])
            self.assertEqual(quality.sha(before), result["sources"][0]["sha256"])
            self.assertNotIn("performance_eligible", row)


if __name__ == "__main__": unittest.main()
