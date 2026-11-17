"""N-gram adapter evidence gates; all process receipts and rows are synthetic."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eng.regression import speculation


def rows():
    plain = {"Scenario": "spec", "Label": "plain greedy", "Finish": "eos", "Note": "",
             "Prompt": 128, "Reused": 0, "Fresh": 128, "OutTokens": 3, "Tokens": [7, 8, 9],
             "PrefillTps": 12.0, "DecodeTps": 5.0, "TtftMs": 10.0,
             "Drafted": 0, "Accepted": 0, "VerifySteps": 0, "PlainSteps": 3, "Rollbacks": 0}
    candidate = dict(plain, Label="ngram", Drafted=3, Accepted=2, VerifySteps=1, PlainSteps=1)
    return [plain, candidate]


class NgramTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.binary = self.folder / "AgentTurnBench.exe"
        self.binary.write_bytes(b"synthetic receipt only - never executed")
        self.model = self.folder / "fixture.gguf"
        self.model.write_bytes(b"synthetic model - never loaded")
        self.output = self.folder / "evidence"
        self.context = {"repo": Path(__file__).resolve().parents[3], "artifact_dir": self.output,
                        "model_path": self.model, "backend": "ggml_cuda", "timeout_seconds": 30,
                        "model_family": "gemma4",
                        "benchmark_command": [str(self.binary)]}
        self.calls = []

    def receipt(self, command, kwargs):
        return {"command": command, "stage": kwargs["stage"], "stdout": "Backend: GgmlCuda\n",
                "stderr": "", "exit_code": 0, "timed_out": False, "output_truncated": False,
                "cancelled": False, "teardown": {"verified": True}, "duration_seconds": .1,
                "stdout_path": str(Path(kwargs["artifact_dir"]) / "ngram.stdout.log"),
                "stderr_path": str(Path(kwargs["artifact_dir"]) / "ngram.stderr.log")}

    def invoke(self, data=None, change=None, write_rows=True):
        def execute(command, **kwargs):
            self.calls.append((command, kwargs))
            if write_rows:
                Path(command[command.index("--out") + 1]).write_text(json.dumps(data if data is not None else rows()), encoding="utf-8")
            item = self.receipt(command, kwargs)
            if change: item.update(change)
            return item
        with patch.object(speculation, "run_process", side_effect=execute):
            return speculation.run_ngram({"id": "fixture-ngram"}, self.context)

    def test_active_exact_parity_and_bounded_command(self):
        self.context["env"] = {"TS_SPEC_DRAFT_MODEL": "must-not-load.gguf", "TS_SPEC_DRAFT": "900",
                               "TS_QWEN35_DFLASH": "must-not-load.gguf", "TS_N_CPU_MOE": "2"}
        result = self.invoke()
        self.assertEqual(result["status"], "pass", result["reason"])
        command, kwargs = self.calls[0]
        for flag, value in (("--scenarios", "spec"), ("--spec-candidates", "ngram"), ("--spec-new", "64"),
                            ("--spec-file", "128"), ("--warmup", "0"), ("--measure-passes", "1")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--spec-only", command)
        self.assertNotIn("--spec-engine", command)
        self.assertNotIn("--draft-model", command)
        self.assertNotIn("TS_SPEC_DRAFT_MODEL", kwargs["env"])
        self.assertNotIn("TS_QWEN35_DFLASH", kwargs["env"])
        self.assertEqual(kwargs["env"]["TS_N_CPU_MOE"], "2")
        self.assertTrue(result["evidence"]["parity"]["activated"])
        self.assertTrue(result["inputs"]["sources"])
        self.assertEqual(result["inputs"]["sampling"]["seed"], -1)
        self.assertEqual(result["processes"][0]["stdout"], "Backend: GgmlCuda\n")
        self.assertTrue(self.output.joinpath("scenario.json").is_file())

    def test_missing_binary_and_contaminating_flags_are_untested(self):
        for command in ([str(self.folder / "missing")], [str(self.binary), "--spec-engine", "auto"], "shell command"):
            with self.subTest(command=command), patch.object(speculation, "run_process") as process:
                self.context["benchmark_command"] = command
                result = speculation.run_ngram({}, self.context)
                self.assertEqual(result["status"], "untested")
                process.assert_not_called()

    def test_missing_model_or_invalid_backend_is_unavailable(self):
        self.context["model_path"] = self.folder / "missing.gguf"
        self.assertFalse(speculation.availability(self.context)[0])
        self.context.update(model_path=self.model, backend="unknown")
        self.assertFalse(speculation.availability(self.context)[0])

    def test_known_family_backend_gate(self):
        for family, allowed, refused in (("gemma4", "cuda", "cpu"), ("muse_spark", "ggml_vulkan", "cuda"),
                                         ("minimax-m3", "ggml_cpu", "ggml_metal")):
            with self.subTest(family=family):
                self.context.update(model_family=family, backend=allowed)
                self.assertTrue(speculation.availability(self.context)[0])
                self.context["backend"] = refused
                self.assertFalse(speculation.availability(self.context)[0])

    def test_missing_and_unknown_family_are_unavailable_without_process(self):
        for family in (None, "", "nemotron_h", "unsupported-family"):
            with self.subTest(family=family), patch.object(speculation, "run_process") as process:
                self.context.pop("model_family", None)
                if family is not None:
                    self.context["model_family"] = family
                ready, reason = speculation.availability(self.context)
                self.assertFalse(ready)
                self.assertIn("source-verified family", reason)
                result = speculation.run_ngram({}, self.context)
                self.assertEqual(result["status"], "untested")
                process.assert_not_called()

    def test_terminal_reason_and_context_bound(self):
        for finish in ("garbage", "length", "stop", "stop_sequence", "repetition", "aborted", "max_tokens"):
            data = rows()
            for row in data: row["Finish"] = finish
            with self.subTest(finish=finish):
                self.assertEqual(self.invoke(data)["status"], "fail")
        data = rows()
        for row in data: row.update(Prompt=1985, Fresh=1985)
        self.assertEqual(self.invoke(data)["status"], "fail")
        data = rows()
        for row in data: row.update(Finish="max_tokens", Tokens=list(range(64)), OutTokens=64)
        self.assertEqual(self.invoke(data)["status"], "pass")

    def test_positive_throughput_only_for_work_that_ran(self):
        for key in ("PrefillTps", "DecodeTps", "TtftMs"):
            for value in (0, -1, "invalid", None, True):
                data = rows(); data[1][key] = value
                with self.subTest(key=key, value=value):
                    self.assertEqual(self.invoke(data)["status"], "fail")
        data = rows()
        for row in data: row.update(Reused=128, Fresh=0, PrefillTps=0, Tokens=[7], OutTokens=1, DecodeTps=0)
        self.assertEqual(self.invoke(data)["status"], "pass")

    def test_zero_activation_is_untested(self):
        data = rows()
        data[1].update(Drafted=0, Accepted=0, VerifySteps=0)
        result = self.invoke(data)
        self.assertEqual(result["status"], "untested")
        self.assertIn("not positively exercised", result["reason"])
        self.assertFalse(result["evidence"]["parity"]["activated"])

    def test_first_divergence_survives_nonzero_benchmark_exit(self):
        data = rows()
        data[1]["Tokens"] = [7, 99, 9]
        result = self.invoke(data, {"exit_code": 1})
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["evidence"]["parity"]["first_divergence"], 1)

    def test_mismatching_tokens_finish_shapes_or_baseline_speculation_fail(self):
        mutations = [lambda d: d[1].update(Tokens=[7, 8, 99]), lambda d: d[1].update(Finish="max_tokens"),
                     lambda d: d[1].update(Reused=1, Fresh=127), lambda d: d[0].update(Drafted=1),
                     lambda d: d[0].update(Rollbacks=1)]
        for mutate in mutations:
            data = rows(); mutate(data)
            with self.subTest(data=data):
                self.assertEqual(self.invoke(data)["status"], "fail")

    def test_duplicate_extra_empty_invalid_rows_and_counter_contracts_fail(self):
        cases = [[], [rows()[0]], rows() + [rows()[1]], rows() + [dict(rows()[1], Label="draft head (auto)")]]
        for key, value in (("Tokens", []), ("Tokens", [True, 8, 9]), ("OutTokens", 99),
                           ("Accepted", 5), ("VerifySteps", -1), ("Drafted", None), ("Prompt", 0),
                           ("Finish", "error"), ("Note", "FAIL: wrong row")):
            data = rows(); data[1][key] = value; cases.append(data)
        for data in cases:
            with self.subTest(data=data):
                self.assertEqual(self.invoke(data)["status"], "fail")

    def test_nonfinite_metrics_anywhere_fail(self):
        for key in ("PrefillTps", "DecodeTps", "TtftMs", "OptionalMetric"):
            data = rows(); data[1][key] = float("nan")
            with self.subTest(key=key):
                result = self.invoke(data)
                self.assertEqual(result["status"], "fail")
                json.loads(self.output.joinpath("scenario.json").read_text(), parse_constant=lambda x: self.fail(x))

    def test_backend_fallback_missing_banner_and_wrong_backend_fail(self):
        for text in ("", "backend=ggml_cuda (requested)", "Backend: Cpu\n",
                     "Backend: GgmlCuda\nbackend falling back to CPU\n",
                     "Backend: GgmlCuda\nBackend: Cpu\n"):
            with self.subTest(text=text):
                self.assertEqual(self.invoke(change={"stdout": text})["status"], "fail")

    def test_timeout_oom_truncation_and_unverified_teardown_fail(self):
        for change in ({"timed_out": True}, {"stderr": "CUDA_ERROR_OUT_OF_MEMORY"},
                       {"stderr": "System.OutOfMemoryException"}, {"output_truncated": True},
                       {"exit_code": -1}, {"teardown": {"verified": False}}, {"cancelled": True}):
            with self.subTest(change=change):
                self.assertEqual(self.invoke(change=change)["status"], "fail")

    def test_stale_output_never_accepted_or_deleted(self):
        self.output.mkdir()
        stale = self.output / "rows.json"
        stale.write_text(json.dumps(rows()))
        self.assertEqual(self.invoke(write_rows=False)["status"], "fail")
        self.assertTrue(stale.is_file())
        first = self.invoke()
        second = self.invoke(write_rows=False)
        self.assertEqual(first["status"], "pass")
        self.assertEqual(second["status"], "fail")
        self.assertNotEqual(first["evidence"]["artifact_dir"], second["evidence"]["artifact_dir"])

    def test_credentials_sanitized_in_result(self):
        secret = "fixture-secret-12345"
        self.context["env"] = {"API_KEY": secret}
        result = self.invoke(change={"stderr": "api_key=" + secret})
        self.assertNotIn(secret, json.dumps(result))
        self.assertNotIn(secret, self.output.joinpath("scenario.json").read_text())

    def test_cancellation_saves_receipt_then_rethrows(self):
        def execute(command, **kwargs):
            receipt = self.receipt(command, kwargs)
            receipt.update(cancelled=True, exit_code=-1)
            Path(kwargs["artifact_dir"]).joinpath("ngram.process.json").write_text(json.dumps(receipt))
            raise KeyboardInterrupt
        with patch.object(speculation, "run_process", side_effect=execute), self.assertRaises(KeyboardInterrupt):
            speculation.run_ngram({}, self.context)
        result = json.loads(self.output.joinpath("scenario.json").read_text())
        self.assertEqual(result["status"], "untested")
        self.assertTrue(result["processes"][0]["teardown"]["verified"])

    def test_cancellation_without_teardown_evidence_is_not_pass(self):
        with patch.object(speculation, "run_process", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            speculation.run_ngram({}, self.context)
        result = json.loads(self.output.joinpath("scenario.json").read_text())
        self.assertEqual(result["status"], "fail")
        self.assertIn("unknown", result["evidence"]["cancellation"])


if __name__ == "__main__":
    unittest.main()
