"""Offline audit admission/truth tests. No network, device or generated-code execution."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "ninfer-api-no-loss-audit.py"
spec = importlib.util.spec_from_file_location("ninfer_no_loss_audit_tests", SOURCE)
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


class TextTruthTests(unittest.TestCase):
    def profile(self):
        header = {key: 1 for key in ("harness_sha256", "scenarios", "repeats", "answer_tokens", "decode_tokens", "nonce_seed", "sampling", "tokenizer_sha256", "haystack", "quality_tasks", "gsm8k_limit", "long_order_quality", "long_prompt_tolerance")}
        expected = {"result": 6, "sorted": [1, 2], "sum": 3, "unique_count": 2}
        body = {"messages": [{"role": "user", "content": "full original input"}], "temperature": 0, "think": False}
        record = {"requests": [{"prompt_tokens": 100, "cached_tokens": 0, "request_sha256": "id", "response_sha256": "response", "content": json.dumps(expected)}],
                  "checks": [{"name": "task0", "ok": False}, {"name": "stream_completed", "ok": True}], "notes": {"expected": [expected]}}
        return {"header": header, "cases": {("quality_tasks", 0): record}, "bodies": {("quality_tasks", 0): [body]}}

    def compare(self, a, b):
        return audit.compare_text(a, b, SimpleNamespace(request_identity=lambda x: x))

    def test_field_truths_inside_failed_whole_task_are_preserved(self):
        a = self.profile(); b = copy.deepcopy(a)
        b["cases"][("quality_tasks", 0)]["requests"][0]["content"] = '{"result":6,"sorted":[1,2],"sum":4,"unique_count":2}'
        result = self.compare(a, b)
        self.assertEqual(4, result["arithmeticOldTrue"])
        self.assertFalse(result["noLossPassed"])
        self.assertEqual("sum", result["arithmeticLostTruths"][0]["field"])

    def test_one_gain_does_not_cancel_one_loss(self):
        a = self.profile(); b = copy.deepcopy(a)
        checks = b["cases"][("quality_tasks", 0)]["checks"]
        checks[0]["ok"] = True; checks[1]["ok"] = False
        result = self.compare(a, b)
        self.assertEqual(1, len(result["originalGains"]))
        self.assertEqual(1, len(result["originalLostTruths"]))
        self.assertFalse(result["noLossPassed"])

    def test_missing_old_true_check_is_not_dropped(self):
        a = self.profile(); b = copy.deepcopy(a)
        b["cases"][("quality_tasks", 0)]["checks"] = b["cases"][("quality_tasks", 0)]["checks"][:1]
        result = self.compare(a, b)
        self.assertFalse(result["noLossPassed"])
        self.assertFalse(result["originalLostTruths"][0]["candidateCheckPresent"])

    def test_changed_generation_setting_or_full_prompt_fails_comparability(self):
        for field, value in (("temperature", 1), ("think", True), ("messages", [{"role": "user", "content": "changed"}])):
            a = self.profile(); b = copy.deepcopy(a); b["bodies"][("quality_tasks", 0)][0][field] = value
            result = self.compare(a, b)
            self.assertFalse(result["comparable"]); self.assertFalse(result["noLossPassed"])
            self.assertEqual("not_comparable", result["originalLostTruths"][0]["transition"])

    def test_prompt_count_change_or_wrong_json_type_is_not_comparable(self):
        for value in (101, 100.0, True, None):
            a = self.profile(); b = copy.deepcopy(a); b["cases"][("quality_tasks", 0)]["requests"][0]["prompt_tokens"] = value
            self.assertFalse(self.compare(a, b)["noLossPassed"])

    def test_string_integer_bool_and_missing_fields_are_not_interchangeable(self):
        for content in ('{"result":"6","sorted":[1,2],"sum":3,"unique_count":2}', '{"result":6,"sorted":[true,2],"sum":3,"unique_count":2}', '{}', '```json\n{}\n```'):
            a = self.profile(); b = copy.deepcopy(a); b["cases"][("quality_tasks", 0)]["requests"][0]["content"] = content
            self.assertFalse(self.compare(a, b)["noLossPassed"])

    def test_cache_hit_count_changes_are_reported_without_changing_logical_inputs(self):
        a = self.profile(); b = copy.deepcopy(a); b["cases"][("quality_tasks", 0)]["requests"][0]["cached_tokens"] = 64
        result = self.compare(a, b)
        self.assertTrue(result["noLossPassed"])
        self.assertEqual(64, result["requests"][0]["candidateCachedTokens"])


class ProgramTruthTests(unittest.TestCase):
    def profile(self):
        fixture = [{"id": 0, "input": {"n": 0}, "expected": []}, {"id": 1, "input": {"n": 3}, "expected": [2]}]
        row = {"passed": False, "request": {"request_body": {"model": "same", "messages": ["full input"]}, "http_status": 200, "error": None, "raw_response_complete": True, "finish_reason": "stop"},
               "evaluation": {"checks": {"finish_reason_stop": True, "complete_module": True, "closed_AST_policy": True, "all_functional_cases": False}}}
        result = {"campaign": {"nonce_seed": 1}, "records": {("primes", 0): row}, "executed": {("primes", 0): {0: {"status": "PASS", "passed": True}, 1: {"status": "WRONG_OUTPUT", "passed": False}}}}
        return result, SimpleNamespace(TASKS=[{"id": "primes", "cases": fixture}])

    def test_positive_case_in_failed_program_is_not_lost_to_a_gain(self):
        a, tool = self.profile(); b = copy.deepcopy(a)
        b["executed"][("primes", 0)] = {0: {"status": "WRONG_OUTPUT", "passed": False}, 1: {"status": "PASS", "passed": True}}
        result = audit.compare_program(a, b, tool)
        self.assertEqual(1, result["functionalOldTrue"]); self.assertFalse(result["noLossPassed"])
        self.assertEqual(0, result["lostFunctionalTruths"][0]["caseId"])

    def test_policy_refusal_or_not_executed_cannot_preserve_a_previous_pass(self):
        a, tool = self.profile(); b = copy.deepcopy(a); b["executed"][("primes", 0)] = {}
        b["records"][("primes", 0)]["evaluation"]["checks"]["closed_AST_policy"] = False
        result = audit.compare_program(a, b, tool)
        self.assertEqual("NOT_EXECUTED", result["lostFunctionalTruths"][0]["candidateStatus"])
        self.assertTrue(result["lostChecks"]); self.assertFalse(result["noLossPassed"])

    def test_wire_model_or_prompt_changes_are_explicitly_incomparable(self):
        for field, value in (("model", "different"), ("messages", ["different"])):
            a, tool = self.profile(); b = copy.deepcopy(a); b["records"][("primes", 0)]["request"]["request_body"][field] = value
            result = audit.compare_program(a, b, tool)
            self.assertFalse(result["comparable"]); self.assertFalse(result["noLossPassed"])

    def test_generation_failure_is_a_lost_truth_even_for_old_failed_program(self):
        a, tool = self.profile(); b = copy.deepcopy(a); b["records"][("primes", 0)]["request"]["http_status"] = 500
        result = audit.compare_program(a, b, tool)
        self.assertFalse(result["noLossPassed"])
        self.assertEqual("completeGeneration", result["lostChecks"][0]["name"])


class AdmissionTests(unittest.TestCase):
    def test_duplicate_or_nonfinite_evidence_container_fields_are_refused(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):
                audit.parse(value)

    def test_duplicate_check_identity_and_nonboolean_truth_are_refused(self):
        with self.assertRaises(ValueError):
            audit.checks_by_name([{"name": "x", "ok": True}, {"name": "x", "ok": False}])
        with self.assertRaises(ValueError):
            audit.checks_by_name([{"name": "x", "ok": 1}])

    def test_incomplete_program_directory_is_not_implicitly_text_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "complete original program"):
                audit.program_path(Path(temporary))

    def test_incomplete_ignis_like_campaign_is_refused_before_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "program"; folder.mkdir()
            (folder / "campaign.json").write_text(json.dumps({"complete": False, "status": "RUNNING_NOT_PASS", "identities_unchanged": True}))
            with self.assertRaisesRegex(ValueError, "Incomplete program campaign"):
                audit.audit_program(Path(temporary), SimpleNamespace(), audit.Evidence(), {})

    def test_original_response_parser_behavior_for_arithmetic_content_is_preserved(self):
        a = TextTruthTests().profile(); b = copy.deepcopy(a)
        b["cases"][("quality_tasks", 0)]["requests"][0]["content"] = '{"result":0,"result":6,"sorted":[1,2],"sum":3,"unique_count":2}'
        self.assertTrue(TextTruthTests().compare(a, b)["noLossPassed"])


if __name__ == "__main__":
    unittest.main()
