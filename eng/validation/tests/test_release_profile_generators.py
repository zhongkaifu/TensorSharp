"""Model-free checks for immutable profile identity and archived workload parity."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from eng.validation.tests.fixtures import deepseek_metadata as FIXTURE

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("deepseek_profiles", SCRIPTS / "make-deepseek-release-profiles.py")
deepseek = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deepseek)


class DeepSeekProfileTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.inputs=FIXTURE.bundle(temporary.name)
        pins=patch.multiple(deepseek,SPARSE_SOURCE=self.inputs['sparse'],REFERENCE_SOURCE=self.inputs['reference'],
                            SPARSE_SHA256=FIXTURE.digest(self.inputs['sparse']),REFERENCE_SHA256=FIXTURE.digest(self.inputs['reference']))
        pins.start();self.addCleanup(pins.stop)

    def profiles(self,pin,sparse=None):
        return deepseek.build_profiles(pin,sparse or self.inputs['sparse'],self.inputs['reference'])

    def command(self,output):
        # Override immutable historical pins only in this test subprocess. The
        # actual CLI argument parser, pin checks and overwrite protection run.
        code=('import importlib.util,sys;sys.path.insert(0,sys.argv[1]);'
              's=importlib.util.spec_from_file_location("fixture_cli",sys.argv[1]+"/make-deepseek-release-profiles.py");'
              'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
              'm.SPARSE_SHA256=sys.argv[2];m.REFERENCE_SHA256=sys.argv[3];'
              'sys.argv=[s.origin,*sys.argv[4:]];m.main()')
        return [sys.executable,'-c',code,str(SCRIPTS),FIXTURE.digest(self.inputs['sparse']),FIXTURE.digest(self.inputs['reference']),
                '--native-sha256','a'*64,'--output',str(output),'--sparse-source-profile',str(self.inputs['sparse']),
                '--decomposed-source-report',str(self.inputs['reference'])]

    def test_full_fixture_workloads_and_launch_settings_are_preserved(self):
        profiles, sources = self.profiles("a" * 64)
        original = deepseek.read_archived(deepseek.SPARSE_SOURCE, deepseek.SPARSE_SHA256)
        reference = deepseek.read_archived(deepseek.REFERENCE_SOURCE, deepseek.REFERENCE_SHA256, report=True)
        self.assertEqual(len(profiles), 3)

        def placement(args):
            flag = next(f for f in ("--tp", "--layer-split") if f in args)
            rest = [a for i, a in enumerate(args) if i not in (args.index(flag), args.index(flag) + 1)]
            return flag, args[args.index(flag) + 1], rest

        def without_tp(env):
            return {key: value for key, value in env.items() if key != "TS_DSV41_TP"}

        _, original_degree, original_rest = placement(original["extra_args"])
        for name, profile in profiles.items():
            self.assertEqual(profile["expected_native_sha256"], "a" * 64)
            self.assertEqual(profile["native_policy"], "exact")
            self.assertEqual(profile['plan_status'],'prepared-not-executed')
            # Layer profiles pass --layer-split; the expert-TP profile passes --tp.
            flag, degree, rest = placement(profile["extra_args"])
            self.assertEqual(flag, "--tp" if name == "accurate-expert-tp7" else "--layer-split")
            self.assertEqual((degree, rest), (original_degree, original_rest))
            self.assertEqual(deepseek.normalized_suites(profile), deepseek.normalized_suites(original))
            self.assertEqual(len(profile["suites"]), 9)
            self.assertEqual(profile["before_suites"], original["before_suites"])
            self.assertEqual(profile["before_suites_quiet_seconds"], original["before_suites_quiet_seconds"])
        self.assertEqual(profiles["accurate-layer7"]["env"], without_tp(original["env"]))
        self.assertEqual(profiles["decomposed-control-layer7"]["env"], without_tp(reference["env"]))
        tp = profiles["accurate-expert-tp7"]
        self.assertEqual(tp["env"], without_tp(original["env"]))
        self.assertEqual(tp["capacity_review"]["max_running_sequences"], 4)
        self.assertEqual(tp["capacity_review"]["max_context"], 65536)
        self.assertEqual(tp["suites"][3][tp["suites"][3].index("--scenarios") + 1], "long_8k,long_32k,long_64k")
        self.assertEqual(tp["suites"][4][tp["suites"][4].index("--concurrency") + 1], "1,4")
        # Per-profile provenance changes must not mutate the shared source manifest.
        self.assertNotIn("rebound_fields", sources["sparse"])

    def test_modified_archive_or_invalid_native_pin_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            tampered = Path(folder) / "profile.json"
            tampered.write_bytes(deepseek.SPARSE_SOURCE.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.profiles("a" * 64, tampered)
        for pin in ("", "A" * 64, "a" * 63, "g" * 64):
            with self.subTest(pin=pin), self.assertRaises(ValueError):
                self.profiles(pin)

    def test_cli_refuses_overwriting_completed_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "plans"
            command = self.command(output)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            before = {f.name: f.read_bytes() for f in output.iterdir()}
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to overwrite", result.stderr)
            self.assertEqual(before, {f.name: f.read_bytes() for f in output.iterdir()})


class ModalityProfileTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.inventory=FIXTURE.bundle(temporary.name)['inventory']

    def command(self, output, baseline="b" * 64, candidate="c" * 64):
        return [sys.executable, str(SCRIPTS / "make-modality-release-profiles.py"),
                "--inventory", str(self.inventory), "--output", str(output),
                "--baseline-sha256", baseline, "--candidate-sha256", candidate]

    def test_every_modality_profile_has_explicit_role_pin_and_cpu_absence(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "plans"
            result = subprocess.run(self.command(output), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["profiles"]), 40)
            self.assertEqual(manifest["pending"], [])
            self.assertEqual(manifest["inventory_sha256"], hashlib.sha256(self.inventory.read_bytes()).hexdigest())
            self.assertEqual(manifest['status'],'prepared-not-executed')
            for entry in manifest["profiles"]:
                profile = json.loads(Path(entry["path"]).read_text())
                expected = "b" * 64 if profile["revision_role"] == "baseline" else "c" * 64
                self.assertEqual(profile["expected_native_sha256"], expected)
                self.assertEqual(profile["native_policy"], "absent" if profile["backend"] == "cpu" else "exact")
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            retry = subprocess.run(self.command(output), capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_invalid_pin_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            for index, bad in enumerate(("", "A" * 64, "g" * 64, "c" * 63)):
                output = Path(folder) / str(index)
                result = subprocess.run(self.command(output, candidate=bad), capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
