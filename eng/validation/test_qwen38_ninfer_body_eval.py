import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("ninfer_eval", Path(__file__).with_name("qwen38-ninfer-body-eval.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NinferEvaluationTests(unittest.TestCase):
    def args(self, suite):
        return SimpleNamespace(suite=suite, python="/python", out=Path("/absolute/artifacts/run"),
                               url="http://127.0.0.1:5088", model="actual-model", engine="tensorsharp",
                               label="body-target-only", tokenizer=Path("/original/frontend/tokenizer.json"),
                               manifest=Path("/absolute/artifacts/server-manifest.json"))

    def test_text_and_short_select_only_original_scenarios(self):
        self.assertEqual((len(MODULE.SUITES["short"]), len(MODULE.SUITES["text"])), (16, 15))
        self.assertEqual(set(MODULE.SUITES["short"]) - set(MODULE.SUITES["text"]), {"image"})
        command = MODULE.build_command(self.args("short"))
        self.assertIn("image", command[command.index("--scenarios") + 1].split(","))

    def test_original_sampling_quality_denominators_and_output_caps_are_explicit(self):
        command = MODULE.build_command(self.args("text"))
        for key, expected in (("--repeats", "3"), ("--quality-tasks", "8"), ("--gsm8k-limit", "8"),
                              ("--temperature", "0"), ("--top-p", "1"), ("--decode-tokens", "1024"),
                              ("--answer-tokens", "512"), ("--nonce-seed", "20261101")):
            self.assertEqual(command[command.index(key) + 1], expected)
        self.assertNotIn("--allow-failures", command)
        self.assertNotIn("--quality-baseline", command)  # Engine-specific baseline semantics are not bypassed.

    def test_long_requests_are_not_replaced_by_shorter_inputs_or_json_repairs(self):
        command = MODULE.build_command(self.args("long"))
        self.assertEqual(command[command.index("--scenarios") + 1], "long_prompt_128k,long_prompt_250k")
        self.assertNotIn("--extra-body", command)
        self.assertIn("--long-order-quality", command)

    def test_program_mode_uses_unchanged_complete_module_protocol(self):
        command = MODULE.build_command(self.args("program"))
        self.assertEqual(command[1], str(MODULE.PROGRAM))
        self.assertIn("--execute", command)
        self.assertNotIn("--decode-tokens", command)
        self.assertNotIn("--baseline", command)

    def test_complete_matrix_preserves_errors_as_quality_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args("long"); args.out = Path(directory)
            rows = [{"kind": "run"}]
            for scenario in MODULE.SUITES["long"]:
                for repeat in range(3):
                    rows.append({"kind": "case", "scenario": scenario, "repeat": repeat,
                                 "verdict": "ERROR", "requests": [{"error": "HTTP500"}]})
            (args.out / "raw.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
            result = MODULE.observed_output(args)
            self.assertTrue(result["complete"])
            self.assertFalse(result["qualityPassed"])
            self.assertEqual(result["verdictCounts"]["ERROR"], 6)

    def test_duplicate_or_missing_case_cannot_be_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args("long"); args.out = Path(directory)
            row = {"kind": "case", "scenario": "long_prompt_128k", "repeat": 0, "verdict": "PASS", "requests": []}
            (args.out / "raw.jsonl").write_text("\n".join(json.dumps(x) for x in [{"kind": "run"}, row, row]))
            result = MODULE.observed_output(args)
            self.assertFalse(result["complete"])
            self.assertFalse(result["qualityPassed"])

    def test_plan_sends_no_http_and_runs_no_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "artifacts/plan"
            argv = ["--engine", "tensorsharp", "--url", "http://127.0.0.1:5088", "--model", "model",
                    "--artifact", "/unavailable/model.ninfer", "--known-artifact-sha", "a" * 64,
                    "--tokenizer", "/unavailable/tokenizer.json", "--manifest", "/unavailable/manifest.json",
                    "--label", "plan", "--out", str(output)]
            with mock.patch.object(MODULE.subprocess, "run", side_effect=AssertionError("No subprocess in PLAN")):
                self.assertEqual(MODULE.main(argv), 0)
            result = json.loads((output / "evaluation.json").read_text())
            self.assertEqual(result["status"], "PLAN_ONLY_NO_HTTP")
            self.assertFalse(result["qualityPassed"])
            self.assertFalse(result["wholeArtifactSha256VerifiedByThisTool"])

    def test_existing_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "artifacts/run"; output.mkdir(parents=True)
            sentinel = output / "evaluation.json"; sentinel.write_text("foreign")
            with self.assertRaisesRegex(ValueError, "Fresh"):
                MODULE.fresh(output)
            self.assertEqual(sentinel.read_text(), "foreign")


if __name__ == "__main__":
    unittest.main()
