"""Exercise the comparison's acceptance and failure gates with small JSON runs."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("compare.py")


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.row = dict(Scenario="spec", Label="ngram", Prompt=100, Reused=20,
                        Fresh=80, OutTokens=3, Tokens=[10, 20, 30], TokenCounts=[],
                        Finish="length", Note="", PrefillTps=100, DecodeTps=50, TtftMs=800)

    def run_comparison(self, baseline=None, candidate=None, repeats=None, extra=(), baseline_repeats=None):
        paths = []
        runs = [baseline if baseline is not None else [self.row],
                candidate if candidate is not None else [self.row]]
        runs.extend(repeats or [])
        runs.extend(baseline_repeats or [])
        for i, rows in enumerate(runs):
            path = Path(self.directory.name) / f"run{i}.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            paths.append(str(path))
        args = [sys.executable, str(SCRIPT), *paths[:2], *extra]
        for i, path in enumerate(paths[2:]):
            flag = "--candidate-repeat" if i < len(repeats or []) else "--baseline-repeat"
            args.extend([flag, path])
        return subprocess.run(args, capture_output=True, text=True, check=False)

    def test_identical_tokens_pass_and_rows_can_be_reordered(self):
        other = dict(self.row, Label="plain")
        result = self.run_comparison([self.row, other], [other, self.row])
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("PASS:", result.stdout)

    def test_each_performance_regression_fails(self):
        for field, value in (("PrefillTps", 94), ("DecodeTps", 47), ("TtftMs", 848)):
            with self.subTest(field=field):
                result = self.run_comparison(candidate=[dict(self.row, **{field: value})])
                self.assertEqual(1, result.returncode)
                self.assertIn("regressed 6.00%", result.stderr)

    def test_threshold_is_configurable_and_inclusive(self):
        result = self.run_comparison(candidate=[dict(self.row, DecodeTps=45)],
                                     extra=["--max-regression-percent", "10"])
        self.assertEqual(0, result.returncode, result.stderr)

    def test_changed_token_fails_even_with_equal_lengths(self):
        result = self.run_comparison(candidate=[dict(self.row, Tokens=[10, 21, 30])])
        self.assertEqual(1, result.returncode)
        self.assertIn("output token 1 changed", result.stderr)

    def test_missing_tokens_and_inconsistent_lengths_fail(self):
        missing = dict(self.row)
        del missing["Tokens"]
        for row in (missing, dict(self.row, Tokens=[])):
            with self.subTest(row=row):
                self.assertEqual(1, self.run_comparison(candidate=[row]).returncode)

    def test_missing_extra_and_duplicate_rows_fail(self):
        other = dict(self.row, Label="other")
        for baseline, candidate in (([self.row, other], [self.row]),
                                    ([self.row], [self.row, other]),
                                    ([self.row], [self.row, self.row])):
            with self.subTest(candidate=candidate):
                self.assertEqual(1, self.run_comparison(baseline, candidate).returncode)

    def test_workload_change_and_reported_failures_fail(self):
        for row in (dict(self.row, Prompt=101, Fresh=81),
                    dict(self.row, Finish="error", Note="ERROR device unavailable"),
                    dict(self.row, Note="info | FAIL: output mismatch"),
                    dict(self.row, Finish="eos/error")):
            with self.subTest(row=row):
                self.assertEqual(1, self.run_comparison(candidate=[row]).returncode)

    def test_malformed_metrics_fail(self):
        for value in (None, "100", float("nan"), float("inf"), -1, True):
            with self.subTest(value=value):
                result = self.run_comparison(candidate=[dict(self.row, DecodeTps=value)])
                self.assertEqual(1, result.returncode)

    def test_first_token_only_has_no_decode_measurement(self):
        row = dict(self.row, OutTokens=1, Tokens=[10], DecodeTps=0)
        result = self.run_comparison([row], [row])
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("n/a", result.stdout)
        self.assertIn("800.00 -> 800.00", result.stdout)

    def test_concurrent_boundaries_are_checked(self):
        row = dict(self.row, TokenCounts=[1, 2], PrefillTps=0)
        self.assertEqual(0, self.run_comparison([row], [row]).returncode)
        for counts in ([2, 1], [1, 1]):
            with self.subTest(counts=counts):
                result = self.run_comparison([row], [dict(row, TokenCounts=counts)])
                self.assertEqual(1, result.returncode)

    def test_unordered_concurrent_token_change_is_informational_unless_required(self):
        row = dict(self.row, TokenCounts=[1, 2], PrefillTps=0)
        changed = dict(row, Tokens=[10, 21, 30])
        result = self.run_comparison([row], [changed])
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("TOKENS INFORMATIONAL:", result.stdout)
        self.assertIn("output token 1 changed", result.stdout)
        self.assertIn("identical tokens except 1 unordered concurrent comparison", result.stdout)
        self.assertNotIn("PASS: identical tokens and workload shapes", result.stdout)
        result = self.run_comparison([row], [changed], extra=["--require-concurrent-identity"])
        self.assertEqual(1, result.returncode)
        self.assertIn("output token 1 changed", result.stderr)

    def test_fixed_arrival_order_and_single_requests_keep_strict_token_identity(self):
        ordered = dict(self.row, TokenCounts=[1, 2], PrefillTps=0, ArrivalOrderFixed=True)
        cases = (
            ("both runs ordered", ordered, dict(ordered, Tokens=[10, 21, 30])),
            ("one request", dict(self.row, TokenCounts=[3]), dict(self.row, TokenCounts=[3], Tokens=[10, 21, 30])),
            ("no boundaries", self.row, dict(self.row, Tokens=[10, 21, 30])),
        )
        for label, before, after in cases:
            with self.subTest(label):
                result = self.run_comparison([before], [after])
                self.assertEqual(1, result.returncode)
                self.assertIn("output token 1 changed", result.stderr)
        # Only one side ordered: the batches may still differ, so it stays informational.
        result = self.run_comparison([ordered], [dict(ordered, ArrivalOrderFixed=False, Tokens=[10, 21, 30])])
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(1, self.run_comparison(candidate=[dict(self.row, ArrivalOrderFixed="yes")]).returncode)

    def test_unordered_concurrent_rows_still_require_request_lengths(self):
        row = dict(self.row, TokenCounts=[1, 2], PrefillTps=0)
        result = self.run_comparison([row], [dict(row, TokenCounts=[2, 1], Tokens=[10, 21, 30])])
        self.assertEqual(1, result.returncode)
        self.assertIn("token counts changed", result.stderr)

    def test_concurrent_first_tokens_only_has_no_decode_measurement(self):
        row = dict(self.row, TokenCounts=[1, 1, 1], DecodeTps=0, PrefillTps=0)
        result = self.run_comparison([row], [row])
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(2, result.stdout.count("n/a"))

    def test_repeated_runs_use_median_but_still_check_every_output(self):
        slow = dict(self.row, DecodeTps=20)
        result = self.run_comparison(candidate=[slow], repeats=[[self.row], [self.row]])
        self.assertEqual(0, result.returncode, result.stderr)
        wrong = copy.deepcopy(self.row)
        wrong["Tokens"][2] = 31
        result = self.run_comparison(repeats=[[self.row], [wrong]])
        self.assertEqual(1, result.returncode)
        self.assertIn("output token 2 changed", result.stderr)

    def timed_row(self, times=(800, 820, 840), total=840):
        return dict(self.row, TokenTimesMs=list(times), TotalMs=total,
                    TtftMs=times[0], PrefillTps=80_000 / times[0],
                    DecodeTps=2000 / (total - times[0]))

    def test_earlier_delivery_and_completion_explain_lower_derived_decode_rate(self):
        # Moving first delivery 20 ms earlier and completion 10 ms earlier
        # increases the measured decode interval from 40 ms to 50 ms.
        before = self.timed_row()
        after = self.timed_row((780, 810, 830), 830)
        result = self.run_comparison([before], [after])
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("50.00 -> 40.00 (-20.00%)", result.stdout)
        self.assertIn("DECODE-RATE EXCEPTION:", result.stdout)
        self.assertIn("all 3 median token delivery times are no later", result.stdout)
        self.assertIn("TotalMs is 840.00 -> 830.00", result.stdout)

    def test_one_later_token_prevents_decode_exception(self):
        result = self.run_comparison([self.timed_row()], [self.timed_row((780, 825, 830), 830)])
        self.assertEqual(1, result.returncode)
        self.assertIn("decode tok/s regressed 20.00%", result.stderr)
        self.assertNotIn("DECODE-RATE EXCEPTION:", result.stdout)

    def test_later_completion_prevents_exception_even_when_every_token_arrives_earlier(self):
        result = self.run_comparison([self.timed_row()], [self.timed_row((780, 810, 830), 850)])
        self.assertEqual(1, result.returncode)
        self.assertIn("decode tok/s regressed", result.stderr)
        self.assertNotIn("DECODE-RATE EXCEPTION:", result.stdout)

    def test_missing_timeline_in_either_side_or_any_repeat_keeps_strict_rate_gate(self):
        before = self.timed_row()
        after = self.timed_row((780, 810, 830), 830)
        missing_before, missing_after = dict(before), dict(after)
        del missing_before["TokenTimesMs"]
        del missing_after["TokenTimesMs"]
        cases = (([missing_before], [after], None, None),
                 ([before], [missing_after], None, None),
                 ([before], [after], [[missing_after]], None),
                 ([before], [after], None, [[missing_before]]))
        for baseline, candidate, repeats, baseline_repeats in cases:
            with self.subTest(case=(baseline, candidate, repeats, baseline_repeats)):
                result = self.run_comparison(baseline, candidate, repeats=repeats,
                                             baseline_repeats=baseline_repeats)
                self.assertEqual(1, result.returncode)
                self.assertIn("decode tok/s regressed", result.stderr)
                self.assertNotIn("DECODE-RATE EXCEPTION:", result.stdout)

    def test_timeline_and_completion_values_are_validated(self):
        before = self.timed_row()
        invalid_times = (None, "800,820,840", [], [800, 820], [800, 840, 820],
                         [-1, 820, 840], [True, 820, 840], [800, float("nan"), 840],
                         [800, 820, float("inf")])
        for times in invalid_times:
            with self.subTest(times=times):
                result = self.run_comparison([before], [dict(before, TokenTimesMs=times)])
                self.assertEqual(1, result.returncode)
                self.assertIn("invalid TokenTimesMs", result.stderr)
        for total in (None, "840", 0, -1, True, float("nan"), float("inf"), 839):
            with self.subTest(total=total):
                result = self.run_comparison([before], [dict(before, TotalMs=total)])
                self.assertEqual(1, result.returncode)
                self.assertIn("invalid TotalMs", result.stderr)
        missing = dict(before)
        del missing["TotalMs"]
        self.assertEqual(1, self.run_comparison([before], [missing]).returncode)

    def test_delivery_exception_uses_medians_and_checks_all_repeated_tokens(self):
        before = self.timed_row()
        after = self.timed_row((780, 810, 830), 830)
        late = self.timed_row((780, 825, 835), 835)
        result = self.run_comparison([before], [late], repeats=[[after], [after]],
                                     baseline_repeats=[[before], [before]])
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("DECODE-RATE EXCEPTION:", result.stdout)
        result = self.run_comparison([before], [after], repeats=[[late], [late]])
        self.assertEqual(1, result.returncode)
        self.assertIn("decode tok/s regressed", result.stderr)
        wrong = dict(after, Tokens=[10, 20, 31])
        self.assertEqual(1, self.run_comparison([before], [after], repeats=[[wrong]]).returncode)

    def test_delivery_exception_does_not_exempt_other_metrics_or_concurrent_aggregates(self):
        before = self.timed_row()
        after = self.timed_row((780, 810, 830), 830)
        result = self.run_comparison([before], [dict(after, PrefillTps=90)])
        self.assertEqual(1, result.returncode)
        self.assertIn("prefill tok/s regressed", result.stderr)
        result = self.run_comparison([dict(before, TokenCounts=[1, 2])],
                                     [dict(after, TokenCounts=[1, 2])])
        self.assertEqual(1, result.returncode)
        self.assertIn("decode tok/s regressed", result.stderr)
        self.assertNotIn("DECODE-RATE EXCEPTION:", result.stdout)


if __name__ == "__main__":
    unittest.main()
