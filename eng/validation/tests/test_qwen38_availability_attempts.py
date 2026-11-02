import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


TOOL = Path(__file__).resolve().parents[1] / "qwen38-availability-attempts.py"
SPEC = importlib.util.spec_from_file_location("availability_attempts", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AvailabilityEvidenceTests(unittest.TestCase):
    def run_audit(self, rows, observed=8, expected=8, exit_code=0):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "external-engine-status").mkdir()
            (root / "raw").mkdir()
            stem = "vllm-plain-fixture"
            (root / "external-engine-status" / (stem + ".json")).write_text(json.dumps({
                "observed_cases": observed, "expected_cases": expected,
                "process_exit_code": exit_code, "quality_passed": True}))
            if rows is not None:
                (root / "raw" / (stem + ".jsonl")).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            output = root / "availability.json"
            with patch("sys.argv", [str(TOOL), "--root", str(root), "--output", str(output), "--engines", "vllm"]), contextlib.redirect_stdout(io.StringIO()):
                MODULE.main()
            return json.loads(output.read_text())

    @staticmethod
    def header(label="fixture"):
        return {"kind": "run", "engine": "vllm", "mode": "plain", "label": label,
                "scenarios": ["fixture_case"], "repeats": 8}

    @staticmethod
    def cases(n=8, passed=True):
        return [{"kind": "case", "scenario": "fixture_case", "repeat": i,
                 "pass": passed, "verdict": "PASS" if passed else "FAIL",
                 "checks": [{"ok": passed}], "requests": [{"finish_reason": "stop"}]}
                for i in range(n)]

    def test_two_incomplete_appended_runs_cannot_combine_into_complete_run(self):
        cases = self.cases(4)
        audit = self.run_audit([self.header(), *cases, self.header(), *cases])
        attempt = next(row for row in audit["missingEngines"] if row.get("label") == "fixture")
        self.assertEqual(attempt["status"], "INCOMPLETE")
        self.assertEqual(attempt["verified_local_raw_cases"], 4)
        self.assertFalse(attempt["quality_passed"])
        self.assertEqual(audit["executedProfiles"], [])

    def test_last_header_identity_must_match_controller(self):
        audit = self.run_audit([self.header("another-label"), *self.cases()])
        attempt = next(row for row in audit["missingEngines"] if row.get("label") == "fixture")
        self.assertEqual(attempt["status"], "INCOMPLETE")
        self.assertFalse(attempt["last_raw_header_identity_matches"])
        self.assertEqual(audit["executedProfiles"], [])

    def test_controller_claim_without_raw_is_not_evidence_of_pass(self):
        audit = self.run_audit(None)
        attempt = next(row for row in audit["missingEngines"] if row.get("label") == "fixture")
        self.assertEqual(attempt["status"], "INCOMPLETE")
        self.assertFalse(attempt["quality_passed"])
        self.assertFalse(attempt["artifacts"]["raw"]["exists_locally"])
        self.assertIsNone(attempt["artifacts"]["raw"]["sha256"])

    def test_warmup_header_with_success_exit_is_not_run(self):
        audit = self.run_audit([self.header()], observed=0, expected=0)
        attempt = next(row for row in audit["missingEngines"] if row.get("label") == "fixture")
        self.assertEqual(attempt["status"], "NOT_RUN")
        self.assertFalse(attempt["quality_passed"])

    def test_complete_formal_case_count_still_does_not_assert_quality_pass(self):
        audit = self.run_audit([self.header(), *self.cases(passed=False)])
        # The controller claim stays available without overriding raw failures.
        self.assertEqual(audit["executedProfiles"][0]["status"], "EXECUTED")
        self.assertTrue(audit["executedProfiles"][0]["controller_quality_passed"])
        self.assertFalse(audit["executedProfiles"][0]["quality_passed"])
        self.assertIn("不是quality PASS", audit["executedProfiles"][0]["reason"])

    def test_successful_complete_matrix_uses_original_quality_rule(self):
        audit = self.run_audit([self.header(), *self.cases()])
        self.assertTrue(audit["executedProfiles"][0]["quality_passed"])

    def test_failed_process_exit_cannot_claim_quality_pass(self):
        audit = self.run_audit([self.header(), *self.cases()], exit_code=1)
        attempt = next(row for row in audit["missingEngines"] if row.get("label") == "fixture")
        self.assertEqual(attempt["status"], "ERROR")
        self.assertFalse(attempt["quality_passed"])

    def test_duplicate_repeat_cannot_replace_expected_repeats(self):
        audit = self.run_audit([self.header(), *[self.cases(1)[0]] * 8])
        attempt = next(row for row in audit["missingEngines"] if row.get("label") == "fixture")
        self.assertEqual(attempt["status"], "INCOMPLETE")
        self.assertFalse(attempt["quality_passed"])

    def test_original_failed_check_overrides_case_and_controller_claim(self):
        cases = self.cases()
        cases[0]["checks"][0]["ok"] = False
        audit = self.run_audit([self.header(), *cases])
        self.assertFalse(audit["executedProfiles"][0]["quality_passed"])

    def test_header_repeat_count_must_be_integer_not_bool(self):
        header = {**self.header(), "repeats": True}
        audit = self.run_audit([header, *self.cases()])
        attempt = next(row for row in audit["missingEngines"] if row.get("label") == "fixture")
        self.assertEqual(attempt["status"], "INCOMPLETE")
        self.assertFalse(attempt["quality_passed"])

    def test_effective_configuration_preserves_prefix_disable_and_last_cpu_budget(self):
        flags = MODULE.effective_flags(["vllm", "serve", "model", "--enable-prefix-caching",
                                        "--cpu-offload-gb", "0", "--no-enable-prefix-caching",
                                        "--cpu-offload-gb=1"])
        self.assertIs(flags["--enable-prefix-caching"], False)
        self.assertEqual(flags["--cpu-offload-gb"], "1")

    def test_native_explicit_device_budget_refusal_is_configuration_unsupported(self):
        line = "2026-11-02 01:27:56.342 FATAL server failed during startup | Device budget cannot hold model weights and required execution resources"
        status, reason, excerpt = MODULE.classify(line)
        self.assertEqual(status, "UNSUPPORTED")
        self.assertIn("该配置", reason)
        self.assertEqual(excerpt, line)
        # An arbitrary planning failure does not establish a capacity limitation.
        self.assertEqual(MODULE.classify("ERROR planning runtime failed")[0], "ERROR")

    def test_native_kv_reservation_and_headroom_refusal_preserves_actual_bytes(self):
        line = "FATAL automatic KV capacity requires 11502815489 bytes total (10429073665 bytes minimum Engine runtime reservation + 1073741824 bytes automatic headroom), but only 10082844672 bytes are available after weights"
        status, _, excerpt = MODULE.classify(line)
        self.assertEqual(status, "UNSUPPORTED")
        self.assertEqual(excerpt, line)
        self.assertEqual(MODULE.classify("automatic KV capacity requires an unsupported allocator")[0], "ERROR")


if __name__ == "__main__":
    unittest.main()
