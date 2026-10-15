import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("freeze_teacher", Path(__file__).parents[1] / "freeze-qwen-teacher-variant.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FreezeTeacherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = self.root / "base"
        app = self.baseline / "bin/Release/net10.0"
        app.mkdir(parents=True)
        (app / "InferenceWeb.Tests.dll").write_bytes(b"frozen fixture")
        (app / "TensorSharp.Models.dll").write_bytes(b"frozen model")
        (app / "GgmlOps.dll").write_bytes(b"old native")
        (self.baseline / "plan.json").write_text(json.dumps({"steps": 24, "environment": {"TS_Q4E_SPAN_TRACE": "1"}, "native_sha256": "old"}))
        (self.baseline / "run-once.py").write_text("raise RuntimeError('must not execute')")
        self.native = self.root / "new.dll"
        self.native.write_bytes(b"new native")
        self.output = self.root / "new"

    def prepare(self, **kwargs):
        return MOD.prepare(self.baseline, self.native, MOD.sha(self.native), kwargs.pop("destination", self.output), self.root, **kwargs)

    def test_native_only_copy_preserves_managed_and_does_not_execute(self):
        report = self.prepare()
        self.assertTrue(report["all_non_native_files_equal"])
        self.assertEqual({}, report["environment_changes"])
        self.assertEqual(b"frozen model", (self.output / "bin/Release/net10.0/TensorSharp.Models.dll").read_bytes())
        self.assertFalse((self.output / "run").exists())

    def test_explicit_diagnostic_environment_is_recorded(self):
        report = self.prepare(environment={"TS_Q4E_ATTN_TRACE_LAYER": "39"})
        self.assertEqual({"before": None, "after": "39"}, report["environment_changes"]["TS_Q4E_ATTN_TRACE_LAYER"])
        self.assertEqual(24, json.loads((self.output / "plan.json").read_text())["steps"])

    def test_existing_destination_is_preserved(self):
        self.output.mkdir()
        (self.output / "sentinel").write_bytes(b"evidence")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(b"evidence", (self.output / "sentinel").read_bytes())

    def test_native_pin_mismatch_creates_no_output(self):
        with self.assertRaises(ValueError):
            MOD.prepare(self.baseline, self.native, "0" * 64, self.output, self.root)
        self.assertFalse(self.output.exists())

    def test_destination_outside_root_or_overlapping_baseline_is_rejected(self):
        for destination in (self.root.parent / "outside", self.baseline / "nested", self.root):
            with self.assertRaises(ValueError):
                self.prepare(destination=destination)

    def test_steps_update_expected_result_and_environment_atomically(self):
        report = self.prepare(steps=64)
        plan = json.loads((self.output / "plan.json").read_text())
        self.assertEqual(64, plan["steps"])
        self.assertEqual("64", plan["environment"]["TS_TEST_QWEN4EXP_BATCH_STEPS"])
        self.assertEqual({"before": 24, "after": 64}, report["plan_changes"]["steps"])

    def test_steps_refuse_invalid_or_conflicting_environment(self):
        for kwargs in ({"steps": 7}, {"steps": 64, "environment": {"TS_TEST_QWEN4EXP_BATCH_STEPS": "24"}}):
            with self.assertRaises(ValueError):
                self.prepare(**kwargs)
            self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
