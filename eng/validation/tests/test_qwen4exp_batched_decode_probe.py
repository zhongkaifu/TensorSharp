"""Exercise provenance and fail-closed result binding without loading models."""
import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("qwen4exp_batch_probe", Path(__file__).parents[1] / "qwen4exp-batched-decode-probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def valid_report(model, native, steps=8):
    reports, trace = [], []
    for width in (2, 3, 4):
        for repetition in (0, 1):
            reports.append({"width": width, "repetition": repetition, "steps": steps, "tokens": width * steps,
                            "max_logit_error": .01, "max_kl": 1e-6, "solo_continuation_error": .01,
                            "serial_decode_seconds": 2, "round_robin_decode_seconds": 3, "batched_decode_seconds": 1,
                            "serial_tokens_per_second": width * steps / 2,
                            "round_robin_tokens_per_second": width * steps / 3,
                            "batched_tokens_per_second": width * steps, "greedy_differences": 0,
                            "round_robin_max_logit_error": .01, "round_robin_max_kl": 1e-6,
                            "round_robin_greedy_differences": 0})
            trace.append({"phase": "serial-prefill-control", "width": width, "repetition": repetition,
                          "max_logit_error": .01, "kl": 1e-6})
            for row in range(width):
                for phase in ("independent-prefill", "solo-continuation", "round-robin-prefill"):
                    trace.append({"phase": phase, "width": width, "repetition": repetition, "row": row,
                                  "max_logit_error": .01, "kl": 1e-6})
                for step in range(steps):
                    trace.append({"phase": "batched-decode", "width": width, "repetition": repetition,
                                  "row": row, "step": step, "max_logit_error": .01, "kl": 1e-6,
                                  "greedy_margin": .1, "serial_argmax": 42, "batched_argmax": 42})
            for step in range(steps):
                for row in range(width):
                    trace.append({"phase": "round-robin-decode", "width": width, "repetition": repetition,
                                  "row": row, "step": step, "max_logit_error": .01, "kl": 1e-6,
                                  "greedy_margin": .1, "serial_argmax": 42, "round_robin_argmax": 42})
    return {"model": str(model), "native_library_path": str(native), "backend": "GgmlMetal", "completed": True,
            "failure": None, "steps_per_repetition": steps, "repetitions": 2, "reports": reports, "trace": trace,
            "round_robin_iteration_order": "step-then-row", "round_robin_cache_bind_in_timing": True}


def write_trx(path, outcome="Passed", executed="1", name=probe.TEST):
    path.write_text(f'<TestRun xmlns="http://microsoft.com/schemas/VisualStudio/TeamTest/2010">'
                    f'<Results><UnitTestResult testName="{name}" outcome="{outcome}"/></Results>'
                    f'<ResultSummary><Counters total="1" executed="{executed}" passed="1" /></ResultSummary></TestRun>')


class Qwen4ExpBatchedDecodeProbeTests(unittest.TestCase):
    def test_long_probe_reserves_capacity_without_changing_short_controls(self):
        for steps, capacity in ((8, 128), (16, 128), (32, 128), (64, 128), (128, 256)):
            with self.subTest(steps=steps):
                overrides = probe.probe_environment_overrides(steps)
                self.assertEqual(overrides["TS_KV_INITIAL_TOKENS"], str(capacity))
                self.assertEqual(overrides["MAX_CONTEXT"], "256")

    def fixture(self, directory):
        root = Path(directory)
        model = root / "model with spaces.gguf"
        model.write_bytes(b"GGUF" + b"weights" * 40)
        native = root / "build" / "libGgmlOps.dylib"
        native.parent.mkdir()
        native.write_bytes(b"compiled-native")
        test = root / "test assembly" / "InferenceWeb.Tests.dll"
        test.parent.mkdir()
        test.write_bytes(b"compiled-test")
        (test.parent / native.name).write_bytes(native.read_bytes())
        (test.parent / "TensorSharp.Models.dll").write_bytes(b"compiled-managed")
        args = SimpleNamespace(model=model, native=native, test_dll=test, upstream=root / "upstream",
                               backend="metal", steps=8, min_decode_speedup=None, min_round_robin_speedup=None)
        return root, args

    def test_complete_numerical_report_and_performance_gate(self):
        with TemporaryDirectory() as directory:
            _, args = self.fixture(directory)
            report = valid_report(args.model, args.native)
            result = probe.verify_probe(report, args, probe.digest_file(args.native))
            self.assertEqual(result["status"], "numerical_passed")
            self.assertEqual(result["performance_status"], "measured_not_gated")
            self.assertEqual(result["round_robin_performance_status"], "measured_not_gated")
            self.assertEqual(len(result["measurements"]), 6)
            self.assertEqual(result["trace_phase_counts"]["round-robin-decode"], 144)
            self.assertEqual(result["trace_phase_counts"]["round-robin-prefill"], 18)
            args.min_decode_speedup = 2.01
            with self.assertRaisesRegex(ValueError, "performance gate"):
                probe.verify_probe(report, args, probe.digest_file(args.native))

    def test_round_robin_gate_remains_separate_from_independent_serial_gate(self):
        with TemporaryDirectory() as directory:
            _, args = self.fixture(directory)
            report = valid_report(args.model, args.native)
            args.min_round_robin_speedup = 2.9
            result = probe.verify_probe(report, args, probe.digest_file(args.native))
            self.assertEqual(result["performance_status"], "measured_not_gated")
            self.assertEqual(result["round_robin_performance_status"], "passed_requested_gate")
            self.assertTrue(all(row["decode_wall_speedup"] == 2 and row["round_robin_decode_wall_speedup"] == 3
                                for row in result["measurements"]))
            args.min_round_robin_speedup = 3.01
            with self.assertRaisesRegex(ValueError, "round-robin decode speedup"):
                probe.verify_probe(report, args, probe.digest_file(args.native))
            args.min_round_robin_speedup = 2.9
            args.min_decode_speedup = 2.01
            with self.assertRaisesRegex(ValueError, "Measured decode speedup"):
                probe.verify_probe(report, args, probe.digest_file(args.native))

    def test_missing_corrupt_or_non_interleaved_round_robin_evidence_rejected(self):
        with TemporaryDirectory() as directory:
            _, args = self.fixture(directory)
            original = valid_report(args.model, args.native)
            def rr(report):
                return next(row for row in report["trace"] if row["phase"] == "round-robin-decode")
            def swap_rr_rows(report):
                indices = [i for i, row in enumerate(report["trace"]) if row["phase"] == "round-robin-decode"]
                report["trace"][indices[0]], report["trace"][indices[1]] = report["trace"][indices[1]], report["trace"][indices[0]]
            changes = [lambda report: report.pop("round_robin_cache_bind_in_timing"),
                       lambda report: report.update(round_robin_iteration_order="row-then-step"),
                       lambda report: report["reports"][0].pop("round_robin_decode_seconds"),
                       lambda report: report["reports"][0].update(round_robin_decode_seconds=float("nan")),
                       lambda report: report["reports"][0].update(round_robin_tokens_per_second=999),
                       lambda report: report["reports"][0].update(round_robin_max_logit_error=.005),
                       lambda report: report["reports"][0].update(round_robin_greedy_differences=1),
                       lambda report: report["trace"].remove(rr(report)),
                       lambda report: rr(report).update(max_logit_error=1),
                       lambda report: rr(report).update(kl=.1),
                       lambda report: rr(report).update(round_robin_argmax=43),
                       lambda report: report["trace"].remove(next(row for row in report["trace"] if row["phase"] == "round-robin-prefill")),
                       swap_rr_rows]
            for index, change in enumerate(changes):
                with self.subTest(index=index):
                    report = copy.deepcopy(original)
                    change(report)
                    with self.assertRaises(ValueError):
                        probe.verify_probe(report, args, probe.digest_file(args.native))

    def test_round_robin_cli_gate_requires_finite_positive_value_and_execution(self):
        with TemporaryDirectory() as directory:
            root, args = self.fixture(directory)
            base = ["--model", str(args.model), "--output-dir", str(root / "artifacts" / "new")]
            with patch.object(probe, "ROOT", root):
                self.assertEqual(probe.parse_args(base + ["--min-round-robin-speedup", "1.05"]).min_round_robin_speedup, 1.05)
                for value in ("0", "-1", "nan", "inf"):
                    with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Round-robin speedup gate"):
                        probe.parse_args(base + ["--min-round-robin-speedup", value])
                with self.assertRaisesRegex(ValueError, "Capture-only"):
                    probe.parse_args(base + ["--capture-only", "--min-round-robin-speedup", "1"])

    def test_false_pass_reports_are_rejected(self):
        with TemporaryDirectory() as directory:
            _, args = self.fixture(directory)
            original = valid_report(args.model, args.native)
            changes = [lambda report: report.update(completed=False),
                       lambda report: report.update(failure="native execution failed"),
                       lambda report: report.update(model="/other.gguf"),
                       lambda report: report.update(backend="GgmlCuda"),
                       lambda report: report.update(native_library_path=None),
                       lambda report: report["reports"].pop(),
                       lambda report: report["reports"][0].update(max_kl=.01),
                       lambda report: report["trace"].pop(),
                       lambda report: report["trace"][3].update(max_logit_error=1),
                       lambda report: next(row for row in report["trace"] if row["phase"] == "batched-decode").update(batched_argmax=43)]
            for index, change in enumerate(changes):
                with self.subTest(index=index):
                    report = copy.deepcopy(original)
                    change(report)
                    with self.assertRaises(ValueError):
                        probe.verify_probe(report, args, probe.digest_file(args.native))

    def test_skipped_wrong_test_or_unexecuted_trx_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "probe.trx"
            write_trx(path)
            self.assertEqual(probe.verify_trx(path)["outcome"], "Passed")
            for outcome, executed, name in (("NotExecuted", "0", probe.TEST), ("Passed", "0", probe.TEST), ("Passed", "1", "wrong")):
                with self.subTest(outcome=outcome, executed=executed, name=name):
                    write_trx(path, outcome, executed, name)
                    with self.assertRaises(ValueError):
                        probe.verify_trx(path)

    def test_capture_rejects_stale_native_and_modified_upstream(self):
        with TemporaryDirectory() as directory:
            _, args = self.fixture(directory)
            (args.test_dll.parent / args.native.name).write_bytes(b"old-native")
            with self.assertRaisesRegex(ValueError, "native copy differs"):
                probe.capture(args, {})
            (args.test_dll.parent / args.native.name).write_bytes(args.native.read_bytes())
            with patch.object(probe, "git_identity", return_value={"clean": False}):
                with self.assertRaisesRegex(ValueError, "unchanged upstream"):
                    probe.capture(args, {})

    def test_model_inventory_samples_only_small_windows_and_requires_all_shards(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "model-00001-of-00002.gguf"
            second = root / "model-00002-of-00002.gguf"
            first.write_bytes(b"GGUF" + b"x" * 200000)
            with self.assertRaisesRegex(ValueError, "Missing model shard"):
                probe.model_metadata(first)
            second.write_bytes(b"GGUF" + b"y" * 200000)
            result = probe.model_metadata(first)
            self.assertFalse(result["full_sha256_computed"])
            self.assertEqual(len(result["shards"]), 2)
            for shard in result["shards"]:
                self.assertEqual(sum(window["bytes"] for window in shard["sample_windows"]), 131072)

    def test_environment_redacts_credentials_but_records_token_capacity(self):
        result = probe.environment_identity({"TS_HF_TOKEN": "secret", "TS_KV_INITIAL_TOKENS": "128", "MAX_CONTEXT": "256", "UNRELATED_KEY": "ignored"})
        self.assertEqual(result["TS_HF_TOKEN"], "<redacted>")
        self.assertEqual(result["TS_KV_INITIAL_TOKENS"], "128")
        self.assertNotIn("UNRELATED_KEY", result)

    def test_capture_only_never_invokes_test_and_incomplete_test_exit_zero_fails(self):
        with TemporaryDirectory() as directory:
            root, args = self.fixture(directory)
            base = ["--model", str(args.model), "--native", str(args.native), "--test-dll", str(args.test_dll), "--steps", "8"]
            capture_output = root / "artifacts" / "identity"
            with patch.object(probe, "ROOT", root), patch.object(probe, "capture", return_value={"native_build": {"sha256": probe.digest_file(args.native)}}), \
                    patch.object(probe, "invoke") as invoke, patch("builtins.print"):
                self.assertEqual(probe.main(base + ["--output-dir", str(capture_output), "--capture-only"]), 0)
                invoke.assert_not_called()
            identity = json.loads((capture_output / "execution.json").read_text())
            self.assertFalse(identity["test_executed"])
            self.assertEqual(identity["status"], "provenance_captured_no_test_executed")
            failed_output = root / "artifacts" / "failed"
            def fake_invoke(argv, environment, log, timeout):
                self.assertEqual(argv[0:3], ["dotnet", "test", str(args.test_dll.resolve())])
                self.assertEqual(environment["TS_TEST_QWEN4EXP_MODEL"], str(args.model.resolve()))
                self.assertNotIn("--no-build", argv)
                log.write_text("Fixture test exited zero, but its report is incomplete")
                write_trx(log.parent / "batch-probe.trx")
                report = valid_report(args.model, args.native)
                report["completed"] = False
                (log.parent / "managed-batch-probe.json").write_text(json.dumps(report))
                return {"argv": argv, "exit_code": 0, "timed_out": False, "elapsed_seconds": 1}
            with patch.object(probe, "ROOT", root), patch.object(probe, "capture", return_value={"native_build": {"sha256": probe.digest_file(args.native)}}), \
                    patch.object(probe, "invoke", side_effect=fake_invoke), patch("builtins.print"):
                self.assertEqual(probe.main(base + ["--output-dir", str(failed_output)]), 1)
            failed = json.loads((failed_output / "execution.json").read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertIn("invocation", failed, failed)
            self.assertEqual(failed["invocation"]["exit_code"], 0)
            self.assertIn("incomplete", failed["failures"][0])


if __name__ == "__main__":
    unittest.main()
