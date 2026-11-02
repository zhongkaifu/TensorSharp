"""CPU generation/publication logic only; fake compilers are not CUDA evidence."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("attention_compiler_ablation_tested", HERE / "build.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def resources(regs=120, stack=0, stores=0, loads=0):
    return ("ptxas info : Compiling entry function 'ts_bench_attn_prefill' for 'sm_120a'\n"
            f"ptxas info : Function properties for ts_bench_attn_prefill\n    {stack} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads\n"
            f"ptxas info : Used {regs} registers\n")


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TOOL.BASE.SOURCE.read_text()

    def test_only_the_explicit_hint_changes_candidate_source(self):
        for variant in (3, 4):
            original = TOOL.BASE.generate(self.source, variant)
            self.assertEqual(TOOL.changed_source(self.source, variant, "no_vectorization"), original)
            self.assertEqual(TOOL.changed_source(self.source, variant, "ptxas_o1"), original)
            hinted = TOOL.changed_source(self.source, variant, "min_blocks2")
            self.assertEqual(hinted.count(TOOL.HINT), 1)
            self.assertEqual(hinted.replace(TOOL.HINT, TOOL.LAUNCH), original)

    def test_baseline_source_and_flags_never_receive_candidate_ablation(self):
        for experiment in TOOL.EXPERIMENTS:
            a = TOOL.compile_command("nvcc", "sm_120a", Path("ARM"), 0, experiment)
            self.assertTrue(all(flag in a for flag in TOOL.BASE_FLAGS))
            self.assertIn("-Xptxas=-v,-warn-spills", a)
            self.assertNotIn("-Xptxas=-v,-warn-spills,-O1", a)
        self.assertEqual(TOOL.flags("no_vectorization")[0], [x for x in TOOL.BASE_FLAGS if x != "--extra-device-vectorization"])
        self.assertEqual(TOOL.flags("ptxas_o1")[1], "-Xptxas=-v,-warn-spills,-O1")
        self.assertEqual(TOOL.flags("min_blocks2"), TOOL.flags(None))

    def test_invalid_variant_experiment_or_missing_anchor_is_refused(self):
        for variant in (0, 1, 2, 5):
            with self.assertRaises(ValueError): TOOL.changed_source(self.source, variant, "min_blocks2")
        with self.assertRaises(ValueError): TOOL.changed_source(self.source, 3, "automatic")
        with patch.object(TOOL.BASE, "generate", return_value="unreviewed changed source"):
            with self.assertRaises(ValueError): TOOL.changed_source(self.source, 3, "min_blocks2")


class PublicationTests(unittest.TestCase):
    def setUp(self):
        parent = TOOL.ROOT / "artifacts/qwen38-regression/attention-compiler-ablation-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def invoke(self, output, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return TOOL.main(["--out", str(output), *extra])

    @staticmethod
    def mock_compiler(regs=120, stack=0, stores=0, loads=0, fail_runner=False, omit_binary=False, fail_compile=False, log=None):
        def run(command, **kwargs):
            if command[:2] == ["git", "check-ignore"]:
                return subprocess.CompletedProcess(command, 0, b"ignored", b"")
            if command[-1] == "--version":
                return subprocess.CompletedProcess(command, 0, "FAKE_UNIT_TEST_COMPILER_VERSION_NOT_CUDA\n", "")
            output = Path(command[command.index("-o") + 1])
            runner = "-cubin" not in command
            if runner and fail_runner or not runner and fail_compile:
                return subprocess.CompletedProcess(command, 1, "", "injected compile failure")
            if not omit_binary:
                output.write_bytes(b"FAKE_UNIT_TEST_OUTPUT_NOT_CUDA")
            return subprocess.CompletedProcess(command, 0, "", log if log is not None else resources(regs, stack, stores, loads))
        return run

    def test_generation_has_six_isolated_arms_unchanged_control_no_authorization(self):
        out = self.root / "generated"
        self.assertEqual(self.invoke(out, "--generate-only"), 0)
        top = json.loads((out / "manifest.json").read_text())
        self.assertEqual(len(top["arms"]), 6)
        self.assertEqual(top["compiler"]["version_status"], "NOT_RUN")
        baseline = TOOL.BASE.generate(TOOL.BASE.SOURCE.read_text(), 0)
        for arm in top["arms"]:
            directory = out / arm["directory"]
            self.assertEqual((directory / "variant-0.cu").read_text(), baseline)
            self.assertFalse((directory / "qualified-variants.txt").exists())
            m = json.loads((directory / "manifest.json").read_text())
            self.assertFalse(m["candidate_offline_qualified"])
            self.assertEqual(m["gpu_authorization"], "NOT_AUTHORIZED")
            self.assertEqual(m["runner_source_sha256"], TOOL.sha((TOOL.FROZEN / "bench.cu").read_bytes()))

    def test_hidden_nvcc_flags_are_rejected_before_output_or_compiler(self):
        for index, name in enumerate(("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")):
            out = self.root / f"hidden{index}"
            with patch.dict(os.environ, {name: "--use_fast_math"}), patch.object(TOOL.subprocess, "run") as compiler:
                with self.assertRaises(SystemExit): self.invoke(out, "--generate-only")
                compiler.assert_not_called()
            self.assertFalse(out.exists())

    def test_failed_version_cannot_authorize_or_compile(self):
        out = self.root / "bad-version"
        calls = []
        def unavailable(command, **kwargs):
            calls.append(command)
            if command[:2] == ["git", "check-ignore"]:
                return subprocess.CompletedProcess(command, 0, b"ignored", b"")
            return subprocess.CompletedProcess(command, 1, "", "injected bad version probe")
        with patch.object(TOOL.subprocess, "run", side_effect=unavailable):
            self.assertEqual(self.invoke(out), 1)
        self.assertFalse(any("-cubin" in command for command in calls))
        self.assertFalse(list(out.rglob("qualified-variants.txt")))

    def test_existing_directory_is_never_touched(self):
        out = self.root / "earlier-resource-failure"
        out.mkdir()
        evidence = out / "manifest.json"
        evidence.write_bytes(b"retain exact older failure")
        with self.assertRaises(SystemExit): self.invoke(out, "--generate-only")
        self.assertEqual(evidence.read_bytes(), b"retain exact older failure")

    def test_only_successful_native_resources_and_runner_publish_marker(self):
        out = self.root / "simulated-build"
        fake = self.mock_compiler()
        saw_runner = []
        def observe(command, **kwargs):
            if "-o" in command:
                arm = Path(command[command.index("-o") + 1]).parent
                self.assertFalse((arm / "qualified-variants.txt").exists())
                if "-cubin" not in command: saw_runner.append(True)
            return fake(command, **kwargs)
        with patch.object(TOOL.subprocess, "run", side_effect=observe):
            self.assertEqual(self.invoke(out, "--variants", "4", "--experiments", "min_blocks2"), 0)
        self.assertEqual(saw_runner, [True])
        arm = out / "variant4-min_blocks2"
        self.assertEqual((arm / "qualified-variants.txt").read_text(), "0\n4\n")
        m = json.loads((arm / "manifest.json").read_text())
        self.assertTrue(m["candidate_offline_qualified"])
        self.assertIn("runner_sha256", m)
        self.assertEqual(m["compiler"]["version_status"], "RECORDED")
        self.assertEqual(m["driver_resources"], "NOT_RUN")

    def test_hint_never_overrides_register_stack_or_spill_gates(self):
        for index, kwargs in enumerate((dict(regs=129), dict(stack=8), dict(stores=4), dict(loads=4))):
            out = self.root / f"rejected{index}"
            with patch.object(TOOL.subprocess, "run", side_effect=self.mock_compiler(**kwargs)):
                self.assertEqual(self.invoke(out, "--variants", "3", "--experiments", "min_blocks2"), 0)
            arm = out / "variant3-min_blocks2"
            self.assertFalse((arm / "qualified-variants.txt").exists())
            m = json.loads((arm / "manifest.json").read_text())
            self.assertFalse(m["candidate_offline_qualified"])
            self.assertEqual(m["variants"][1]["status"], "resource_rejected_no_gpu_run")

    def test_failed_runner_compiler_or_missing_output_cannot_publish(self):
        for index, kwargs in enumerate((dict(fail_runner=True), dict(fail_compile=True), dict(omit_binary=True), dict(log="missing ptxas resources"))):
            out = self.root / f"failed{index}"
            with patch.object(TOOL.subprocess, "run", side_effect=self.mock_compiler(**kwargs)):
                self.assertEqual(self.invoke(out, "--variants", "4", "--experiments", "ptxas_o1"), 1)
            self.assertFalse((out / "variant4-ptxas_o1/qualified-variants.txt").exists())

    def test_unavailable_compiler_never_publishes(self):
        def missing(command, **kwargs):
            if command[:2] == ["git", "check-ignore"]:
                return subprocess.CompletedProcess(command, 0, b"ignored", b"")
            raise FileNotFoundError("unit-test missing compiler")
        out = self.root / "no-compiler"
        with patch.object(TOOL.subprocess, "run", side_effect=missing):
            self.assertEqual(self.invoke(out, "--variants", "3", "--experiments", "no_vectorization"), 1)
        self.assertFalse((out / "variant3-no_vectorization/qualified-variants.txt").exists())


if __name__ == "__main__":
    unittest.main()
