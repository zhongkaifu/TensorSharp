"""Jev adapter regressions use synthetic bodies only; no model/server is loaded."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression import jev

REPO = Path(__file__).resolve().parents[3]
MODEL = "synthetic-diffusiongemma.gguf"


def response(schema, expected):
    answers = {}
    for name, question in schema.items():
        kind = question["type"]
        if kind == "noul":
            answer = {"type": kind, "noul": float(expected[name])}
        else:
            labels = list(question["criteria"]) if kind == "choice" else [str(i) for i in range(len(question["criteria"]))]
            winner = expected[name] if kind == "choice" else str(expected[name])
            answer = {"type": kind, "probabilities": {label: float(label == winner) for label in labels}, "confidence": 1.0}
            if kind == "choice":
                answer["choice"] = winner
            else:
                answer.update(score=float(expected[name]), legend={str(i): value for i, value in enumerate(question["criteria"])})
        answers[name] = answer
    return {"model": MODEL, "answers": answers, "usage": {"input_tokens": 42, "output_tokens": 3},
            "diagnostics": {"engine": "tensorsharp", "steps": 1, "seed": 42,
                            "samples": {"policy": "fixed", "n": 1},
                            "probability_semantics": "conditional_label_softmax_temperature_1"}}


class FakeHttp:
    def __init__(self, suite, mutate=None, invalid_status=422):
        self.suite, self.mutate, self.invalid_status = suite, mutate, invalid_status
        self.calls = []

    def json(self, route, payload):
        self.calls.append((route, copy.deepcopy(payload)))
        for case in self.suite["cases"]:
            if payload.get("state") == case["state"]:
                body = response(self.suite["schemas"][case["schema"]], case["expected"])
                if self.mutate:
                    self.mutate(body, len(self.calls))
                return body
        if self.invalid_status == 200:
            return {"accepted": True}
        raise urllib.error.HTTPError("http://127.0.0.1/v1/systemone", self.invalid_status, "synthetic", {}, None)


class JevTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite, cls.fixture_hash = jev.load_fixtures(REPO)
        cls.contract = jev.load_contract(REPO)

    def setUp(self):
        self.result = {"assertions": [], "inputs": {}, "evidence": {}}
        self.context = {"repo": REPO, "model_path": Path(MODEL)}
        self.case = self.suite["cases"][0]
        self.schema = self.suite["schemas"][self.case["schema"]]
        self.payload = self.contract.make_payload(self.case, self.schema, MODEL, 1, 0)

    def validate(self, body):
        return jev.validate_decisions(self.contract, self.payload, body, self.case["expected"], MODEL)

    def test_all_fixed_cases_replay_and_validation_errors_are_executed(self):
        http = FakeHttp(self.suite)
        jev.run_decisions(http, MODEL, self.context, self.result)
        evidence = self.result["evidence"]["jev"]
        self.assertEqual(evidence["counts"], {"fixtures": 12, "passed_fixtures": 12, "decisions": 36, "failures": 0})
        self.assertEqual(len(http.calls), 17)
        self.assertTrue(all(item["passed"] for item in self.result["assertions"]))
        self.assertEqual(evidence["fixture_sha256"], self.fixture_hash)
        self.assertEqual(evidence["seeded_replay"]["max_absolute_probability_delta"], 0)
        self.assertEqual({route for route, _ in http.calls}, {"/v1/systemone"})

    def test_nonfinite_unnormalized_or_wrong_criterion_probabilities_fail(self):
        for probabilities in ({"billing": float("nan"), "technical": 0, "sales": 0},
                              {"billing": 0.6, "technical": 0.1, "sales": 0.1},
                              {"billing": 1, "missing": 0, "sales": 0}):
            body = response(self.schema, self.case["expected"])
            body["answers"]["department"]["probabilities"] = probabilities
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                self.validate(body)

    def test_wrong_gold_decision_cannot_pass_contract_validation(self):
        wrong = dict(self.case["expected"], department="technical")
        body = response(self.schema, wrong)
        with self.assertRaisesRegex(ValueError, "disagrees with expected"):
            self.validate(body)

    def test_score_must_be_finite_weighted_mean_and_legend_order_exact(self):
        for change in ({"score": float("inf")}, {"score": 1}, {"legend": {"0": "wrong", "1": "other", "2": "last"}}):
            body = response(self.schema, self.case["expected"])
            body["answers"]["frustration"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(body)

    def test_request_ownership_usage_and_model_identity_are_verified(self):
        for field in ("keys", "usage", "model"):
            body = response(self.schema, self.case["expected"])
            if field == "keys":
                body["answers"]["other-request"] = body["answers"].pop("urgent")
            elif field == "usage":
                body["usage"]["input_tokens"] = True
            else:
                body["model"] = "another.gguf"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(body)

    def test_ignored_seed_or_sample_controls_and_unknown_semantics_fail(self):
        for change in ({"seed": 0}, {"steps": 2}, {"steps": True}, {"engine": "external"},
                       {"samples": {"policy": "auto", "n": None}}, {"samples": {"policy": "fixed", "n": True}},
                       {"probability_semantics": "unknown"}):
            body = response(self.schema, self.case["expected"])
            body["diagnostics"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(body)

    def test_incorrect_output_records_case_and_continues_other_fixtures(self):
        def mutate(body, call):
            if call == 1:
                body["answers"]["urgent"]["noul"] = 1.0
        http = FakeHttp(self.suite, mutate)
        with self.assertRaisesRegex(ValueError, "billing-calm"):
            jev.run_decisions(http, MODEL, self.context, self.result)
        rows = self.result["evidence"]["jev"]["cases"]
        self.assertEqual(len(rows), 12)
        self.assertFalse(rows[0]["passed"])
        self.assertTrue(all(row["passed"] for row in rows[1:]))

    def test_valid_but_changed_seeded_probabilities_fail_replay(self):
        def mutate(body, call):
            if call == 13:
                body["answers"]["urgent"]["noul"] = 0.1
        with self.assertRaisesRegex(ValueError, "replay probability delta"):
            jev.run_decisions(FakeHttp(self.suite, mutate), MODEL, self.context, self.result)
        self.assertFalse(self.result["evidence"]["jev"]["seeded_replay"]["passed"])

    def test_malformed_accepted_or_server_error_cannot_pass(self):
        for status in (200, 404, 500):
            result = {"assertions": [], "inputs": {}, "evidence": {}}
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "expected validation 4xx"):
                jev.run_decisions(FakeHttp(self.suite, invalid_status=status), MODEL, self.context, result)
            self.assertEqual(len(result["evidence"]["jev"]["malformed_requests"]), 4)

    def test_missing_responses_cannot_be_reported_as_zero_execution_pass(self):
        def mutate(body, call):
            body["answers"] = {}
        with self.assertRaises(ValueError):
            jev.run_decisions(FakeHttp(self.suite, mutate), MODEL, self.context, self.result)
        self.assertEqual(self.result["evidence"]["jev"]["counts"]["passed_fixtures"], 0)
        self.assertTrue(any(item["name"] == "Jev seeded probability replay" and not item["passed"] for item in self.result["assertions"]))

    def test_duplicate_and_empty_fixture_inventory_refused(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / jev._FIXTURE_PATH
            path.parent.mkdir(parents=True)
            for cases in ([], [self.case, self.case]):
                path.write_text(json.dumps({"schemas": self.suite["schemas"], "cases": cases}), encoding="utf-8")
                with self.subTest(count=len(cases)), self.assertRaises(ValueError):
                    jev.load_fixtures(root)


if __name__ == "__main__":
    unittest.main()
