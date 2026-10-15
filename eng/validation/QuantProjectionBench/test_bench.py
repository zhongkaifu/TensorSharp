# Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
"""Controller-only tests. They neither build native code nor use devices/models."""
import argparse
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("quant_projection_bench_controller", Path(__file__).with_name("bench.py"))
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def case(self, name, owned=(1.0, 2.0, 3.0), ordinary=(4.0, 5.0, 6.0)):
        root = self.root / name
        (root / "case").mkdir(parents=True)
        payloads = {"owned.f32le.bin": struct.pack("<" + "f" * len(owned), *owned),
                    "ordinary.f32le.bin": struct.pack("<" + "f" * len(ordinary), *ordinary),
                    "weights.q5_k.bin": b"same weights", "input.f32le.bin": b"same input"}
        for filename, payload in payloads.items():
            (root / "case" / filename).write_bytes(payload)
        report = {"schema": BENCH.SCHEMA, "passed": True,
                  "measurement": {"schema": "tensorsharp.quant-projection-bench.v1", "k": 256, "rows": 3, "columns": 1, "backend": "cuda", "device": 0},
                  "binaries": {filename: BENCH.pin(root / "case" / filename) for filename in payloads}}
        BENCH.write_fresh(root / "receipt.json", report)
        return root

    def change_receipt(self, root, mutate):
        report = BENCH.read(root / "receipt.json")
        mutate(report)
        (root / "receipt.json").write_text(json.dumps(report), encoding="utf-8")

    def measurement(self):
        args = argparse.Namespace(k=256, rows=3, columns=1, backend="cuda", oracle="full", repeats=2,
                                  device=0, warmup=5, iterations=10, sample_rows=16, sample_columns=16)
        value = {"schema": "tensorsharp.quant-projection-bench.v1", "execution_passed": True,
                 "k": 256, "rows": 3, "columns": 1, "backend": "cuda", "oracle": "full",
                 "device": 0, "warmup": 5, "repeats": 2, "iterations_per_repeat": 10,
                 "allocation_upload_timed": False, "source_bytes_unchanged": True,
                 "gpu_event_ms": None, "ordinary_is_numerical_oracle": False,
                 "oracle_complete_outputs": True, "oracle_values": 3, "owned_oracle_passed": True,
                 "variants": {"owned": {"wall_samples_ms": [1.0, 2.0]},
                              "ordinary_performance_reference": {"wall_samples_ms": [3.0, 4.0]}}}
        return args, value

    def test_complete_identical_comparison(self):
        a, b = self.case("a"), self.case("b", ordinary=(99.0, 98.0, 97.0))
        report = BENCH.compare_cases(a, b)
        self.assertTrue(report["bit_identical"])
        self.assertEqual(report["complete_values"], 3)
        self.assertFalse(report["model_quality_performance_hardcap_claim"])

    def test_comparison_missing_binary_digest_rejected(self):
        for filename in ("owned.f32le.bin", "ordinary.f32le.bin", "weights.q5_k.bin", "input.f32le.bin"):
            with self.subTest(filename=filename):
                a, b = self.case("a-" + filename), self.case("b-" + filename)
                self.change_receipt(b, lambda r: r["binaries"].pop(filename))
                with self.assertRaisesRegex(ValueError, "comparison binary digests"):
                    BENCH.compare_cases(a, b)

    def test_comparison_extra_binary_digest_rejected(self):
        a, b = self.case("a"), self.case("b")
        self.change_receipt(b, lambda r: r["binaries"].update({"extra.bin": {"bytes": 0, "sha256": "fake"}}))
        with self.assertRaisesRegex(ValueError, "comparison binary digests"):
            BENCH.compare_cases(a, b)

    def test_comparison_receipt_schema_rejected(self):
        for schema in (None, "wrong-schema"):
            with self.subTest(schema=schema):
                a, b = self.case("a" + str(schema)), self.case("b" + str(schema))
                self.change_receipt(b, lambda r: r.update(schema=schema))
                with self.assertRaisesRegex(ValueError, "receipt/measurement schema"):
                    BENCH.compare_cases(a, b)

    def test_comparison_measurement_schema_rejected(self):
        a, b = self.case("a"), self.case("b")
        self.change_receipt(b, lambda r: r["measurement"].update(schema="wrong-schema"))
        with self.assertRaisesRegex(ValueError, "receipt/measurement schema"):
            BENCH.compare_cases(a, b)

    def test_signed_zero_is_bit_change(self):
        a = self.case("a", owned=(0.0, 2.0, 3.0))
        b = self.case("b", owned=(-0.0, 2.0, 3.0))
        report = BENCH.compare_cases(a, b)
        self.assertFalse(report["bit_identical"])
        self.assertEqual(report["changed_bits_values"], 1)
        self.assertEqual(report["max_abs"], 0)

    def test_tampered_output_rejected(self):
        a, b = self.case("a"), self.case("b")
        (b / "case/owned.f32le.bin").write_bytes(struct.pack("<fff", 7, 8, 9))
        with self.assertRaisesRegex(ValueError, "Retained binary changed"):
            BENCH.compare_cases(a, b)

    def test_different_input_rejected(self):
        a, b = self.case("a"), self.case("b")
        (b / "case/input.f32le.bin").write_bytes(b"different input")
        self.change_receipt(b, lambda r: r["binaries"].update({"input.f32le.bin": BENCH.pin(b / "case/input.f32le.bin")}))
        with self.assertRaisesRegex(ValueError, "Compared sources differ"):
            BENCH.compare_cases(a, b)

    def test_geometry_mismatch_rejected(self):
        a, b = self.case("a"), self.case("b")
        self.change_receipt(b, lambda r: r["measurement"].update(rows=4))
        with self.assertRaisesRegex(ValueError, "geometry/device mismatch"):
            BENCH.compare_cases(a, b)

    def test_incomplete_outputs_rejected(self):
        a, b = self.case("a", owned=(1, 2)), self.case("b", owned=(1, 2))
        with self.assertRaisesRegex(ValueError, "Incomplete output comparison"):
            BENCH.compare_cases(a, b)

    def test_nonfinite_outputs_rejected(self):
        a, b = self.case("a", owned=(float("nan"), 2, 3)), self.case("b")
        with self.assertRaisesRegex(ValueError, "Nonfinite compared output"):
            BENCH.compare_cases(a, b)

    def test_failed_receipt_rejected(self):
        a, b = self.case("a"), self.case("b")
        self.change_receipt(b, lambda r: r.update(passed=False, device_unavailable=True))
        with self.assertRaisesRegex(ValueError, "Both execution receipts"):
            BENCH.compare_cases(a, b)

    def test_full_measurement(self):
        args, value = self.measurement()
        BENCH.validate_measurement(value, args)

    def test_missing_full_oracle_values(self):
        args, value = self.measurement(); value["oracle_values"] = 2
        with self.assertRaisesRegex(ValueError, "oracle value count"):
            BENCH.validate_measurement(value, args)

    def test_oracle_failure_rejected(self):
        args, value = self.measurement(); value["owned_oracle_passed"] = False
        with self.assertRaisesRegex(ValueError, "oracle failed"):
            BENCH.validate_measurement(value, args)

    def test_no_oracle_never_counted_as_pass(self):
        args, value = self.measurement(); args.oracle = value["oracle"] = "none"
        value.update(oracle_values=0, oracle_complete_outputs=False, owned_oracle_passed=None)
        BENCH.validate_measurement(value, args)
        value["owned_oracle_passed"] = True
        with self.assertRaisesRegex(ValueError, "Absent oracle"):
            BENCH.validate_measurement(value, args)

    def test_unrelated_stream_event_claim_rejected(self):
        args, value = self.measurement(); value["gpu_event_ms"] = 1.0
        with self.assertRaisesRegex(ValueError, "GPU-event"):
            BENCH.validate_measurement(value, args)

    def test_nonpositive_timing_rejected(self):
        args, value = self.measurement(); value["variants"]["owned"]["wall_samples_ms"][1] = 0
        with self.assertRaisesRegex(ValueError, "timing samples"):
            BENCH.validate_measurement(value, args)

    def test_verify_rejects_dirty_upstream(self):
        path = self.root / "input"; path.write_bytes(b"frozen")
        expected = {str(path): BENCH.pin(path)["sha256"]}
        with mock.patch.object(BENCH, "git_state", return_value={"revision": "abc", "status": " M file"}):
            with self.assertRaisesRegex(ValueError, "cleanliness"):
                BENCH.verify(expected, self.root, "abc")

    def test_verify_rejects_changed_input(self):
        path = self.root / "input"; path.write_bytes(b"frozen")
        expected = {str(path): BENCH.pin(path)["sha256"]}; path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Input bytes changed"):
            BENCH.verify(expected, self.root, "abc")

    def test_write_fresh_refuses_overwrite(self):
        path = self.root / "report.json"; BENCH.write_fresh(path, {"a": 1})
        with self.assertRaises(FileExistsError):
            BENCH.write_fresh(path, {"a": 2})

    def controller_fixture(self):
        root = self.root / "prepared"
        (root / "source").mkdir(parents=True)
        frozen = root / "source/bench.py"
        actual = self.root / "executing-controller.py"
        frozen.write_bytes(b"frozen controller")
        actual.write_bytes(frozen.read_bytes())
        prepared = {"driver": {"bench.py": BENCH.pin(frozen)}}
        path = root / "prepared.json"
        path.write_text(json.dumps(prepared), encoding="utf-8")
        return prepared, path, actual, frozen

    def test_executing_controller_pin_recorded(self):
        prepared, path, actual, _ = self.controller_fixture()
        with mock.patch.object(BENCH, "__file__", str(actual)):
            identity = BENCH.controller_identity(prepared, path)
        self.assertEqual(identity["path"], str(actual.resolve()))
        self.assertEqual(identity["pin"], prepared["driver"]["bench.py"])

    def test_mutated_executing_controller_rejected(self):
        prepared, path, actual, _ = self.controller_fixture()
        actual.write_bytes(b"mutated executing controller")
        with mock.patch.object(BENCH, "__file__", str(actual)):
            with self.assertRaisesRegex(ValueError, "Executing controller differs"):
                BENCH.controller_identity(prepared, path)

    def test_mutated_frozen_controller_rejected(self):
        prepared, path, actual, frozen = self.controller_fixture()
        frozen.write_bytes(b"mutated frozen controller")
        with mock.patch.object(BENCH, "__file__", str(actual)):
            with self.assertRaisesRegex(ValueError, "frozen controller changed"):
                BENCH.controller_identity(prepared, path)

    def test_build_rejects_controller_mutation_at_entry(self):
        prepared, path, actual, _ = self.controller_fixture()
        prepared["schema"] = BENCH.SCHEMA
        path.write_text(json.dumps(prepared), encoding="utf-8")
        actual.write_bytes(b"mutated actual")
        with mock.patch.object(BENCH, "__file__", str(actual)), mock.patch.object(BENCH, "verify") as verify:
            with self.assertRaisesRegex(ValueError, "Executing controller differs"):
                BENCH.build_driver(argparse.Namespace(prepared=str(path), timeout=1))
            verify.assert_not_called()

    def test_run_rejects_controller_mutation_at_entry(self):
        prepared, path, actual, _ = self.controller_fixture()
        prepared["schema"] = BENCH.SCHEMA
        path.write_text(json.dumps(prepared), encoding="utf-8")
        actual.write_bytes(b"mutated actual")
        with mock.patch.object(BENCH, "__file__", str(actual)), mock.patch.object(BENCH, "verify") as verify:
            with self.assertRaisesRegex(ValueError, "Executing controller differs"):
                BENCH.run_case(argparse.Namespace(prepared=str(path)))
            verify.assert_not_called()

    def test_executing_controller_rechecked_after_mutation(self):
        prepared, path, actual, _ = self.controller_fixture()
        with mock.patch.object(BENCH, "__file__", str(actual)):
            BENCH.controller_identity(prepared, path)
            actual.write_bytes(b"changed after first check")
            with self.assertRaisesRegex(ValueError, "Executing controller differs"):
                BENCH.controller_identity(prepared, path)

    def test_missing_or_extra_variants_rejected(self):
        for variant, remove in (("owned", True), ("ordinary_performance_reference", True), ("extra", False)):
            with self.subTest(variant=variant):
                args, value = self.measurement()
                if remove:
                    del value["variants"][variant]
                else:
                    value["variants"][variant] = {"wall_samples_ms": [1, 2]}
                with self.assertRaisesRegex(ValueError, "timing variants"):
                    BENCH.validate_measurement(value, args)

    def test_device_and_timing_arguments_bound(self):
        for key in ("device", "warmup", "repeats", "iterations_per_repeat"):
            with self.subTest(key=key):
                args, value = self.measurement(); value[key] += 1
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    BENCH.validate_measurement(value, args)

    def test_boolean_metadata_not_accepted_as_integer(self):
        args, value = self.measurement(); args.device = 1; value["device"] = True
        with self.assertRaisesRegex(ValueError, "mismatch"):
            BENCH.validate_measurement(value, args)

    def test_boolean_timing_not_accepted_as_number(self):
        args, value = self.measurement(); value["variants"]["owned"]["wall_samples_ms"] = [True, 1]
        with self.assertRaisesRegex(ValueError, "timing samples"):
            BENCH.validate_measurement(value, args)

    def test_incomplete_or_excessive_sample_count_rejected(self):
        for count in (2, 4):
            with self.subTest(count=count):
                args, value = self.measurement(); args.oracle = value["oracle"] = "sample"
                value.update(oracle_values=count, oracle_complete_outputs=False)
                with self.assertRaisesRegex(ValueError, "oracle value count"):
                    BENCH.validate_measurement(value, args)

    def test_exhaustive_small_sample_completeness_derived(self):
        args, value = self.measurement(); args.oracle = value["oracle"] = "sample"
        value["oracle_complete_outputs"] = False
        coverage = BENCH.validate_measurement(value, args)
        self.assertEqual(coverage["expected_oracle_values"], 3)
        self.assertTrue(coverage["oracle_verified_complete_outputs"])

    def test_partial_sample_completeness_derived(self):
        args, value = self.measurement(); args.oracle = value["oracle"] = "sample"
        args.rows = value["rows"] = 100
        value.update(oracle_values=16, oracle_complete_outputs=False)
        coverage = BENCH.validate_measurement(value, args)
        self.assertFalse(coverage["oracle_verified_complete_outputs"])

    def test_none_oracle_nonzero_count_rejected(self):
        args, value = self.measurement(); args.oracle = value["oracle"] = "none"
        value.update(oracle_values=1, oracle_complete_outputs=False, owned_oracle_passed=None)
        with self.assertRaisesRegex(ValueError, "oracle value count"):
            BENCH.validate_measurement(value, args)

    def test_false_completeness_declaration_rejected(self):
        args, value = self.measurement(); args.oracle = value["oracle"] = "sample"
        value["oracle_complete_outputs"] = True
        with self.assertRaisesRegex(ValueError, "completeness declaration"):
            BENCH.validate_measurement(value, args)

    def test_empty_or_nonfinite_timing_rejected(self):
        for samples in ([], [float("nan"), 1], [float("inf"), 1]):
            with self.subTest(samples=samples):
                args, value = self.measurement(); value["variants"]["owned"]["wall_samples_ms"] = samples
                with self.assertRaisesRegex(ValueError, "timing samples"):
                    BENCH.validate_measurement(value, args)

    def test_build_timeout_cleanup_limitation_recorded(self):
        prepared, path, actual, _ = self.controller_fixture()
        prepared.update(schema=BENCH.SCHEMA, inputs={}, upstream=str(self.root), upstream_revision="abc",
                        configure_argv=["cmake", "fake-configure"], build_argv=["cmake", "fake-build"])
        path.write_text(json.dumps(prepared), encoding="utf-8")
        with mock.patch.object(BENCH, "__file__", str(actual)), mock.patch.object(BENCH, "verify", return_value={}), \
                mock.patch.object(BENCH.subprocess, "run", side_effect=subprocess.TimeoutExpired(["cmake"], 1)):
            self.assertEqual(BENCH.build_driver(argparse.Namespace(prepared=str(path), timeout=1)), 1)
        report = BENCH.read(path.parent / "build-receipt.json")
        self.assertFalse(report["passed"])
        self.assertTrue(report["timeout_direct_child_reaped"])
        self.assertIsNone(report["compiler_descendants_reaped"])
        self.assertIn("not contained", report["timeout_cleanup_limitation"])
        self.assertEqual(report["actual_controller_entry"], report["actual_controller_before"])
        self.assertEqual(report["actual_controller_before"], report["actual_controller_after"])

    def test_build_rechecks_controller_after_work(self):
        prepared, path, actual, _ = self.controller_fixture()
        prepared.update(schema=BENCH.SCHEMA, inputs={}, upstream=str(self.root), upstream_revision="abc",
                        configure_argv=["cmake", "fake-configure"], build_argv=["cmake", "fake-build"])
        path.write_text(json.dumps(prepared), encoding="utf-8")

        def mutate_and_timeout(*args, **kwargs):
            actual.write_bytes(b"mutated while compiler child was executing")
            raise subprocess.TimeoutExpired(["cmake"], 1)

        with mock.patch.object(BENCH, "__file__", str(actual)), mock.patch.object(BENCH, "verify", return_value={}), \
                mock.patch.object(BENCH.subprocess, "run", side_effect=mutate_and_timeout):
            self.assertEqual(BENCH.build_driver(argparse.Namespace(prepared=str(path), timeout=1)), 1)
        report = BENCH.read(path.parent / "build-receipt.json")
        self.assertFalse(report["passed"])
        self.assertIn("Executing controller differs", report["input_bookend_error"])
        self.assertNotEqual(report["actual_controller_entry"]["pin"], report["actual_controller_after_observed"]["pin"])


if __name__ == "__main__":
    unittest.main()
