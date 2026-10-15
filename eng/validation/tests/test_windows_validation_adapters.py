#!/usr/bin/env python3
"""CPU-only launch, shell and independent artifact-oracle checks."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


DIRECTORY = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), DIRECTORY / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VISION = load("qwen38-cuda-vision")
WORKFLOWS = load("validate-release-agent-workflows")
VERIFIER = load("verify-agent-code-artifacts")


class PlatformAdapters(unittest.TestCase):
    def test_dll_launch_preserves_paths_and_does_not_use_a_shell(self):
        path = Path("directory with spaces/TensorSharp.Cli.DLL")
        self.assertEqual(["custom dotnet", str(path)], VISION.cli_launcher(path, "custom dotnet"))
        executable = Path("directory with spaces/TensorSharp.Cli.exe")
        self.assertEqual([str(executable)], VISION.cli_launcher(executable))

    def test_native_deployment_identity_uses_target_platform_filename(self):
        for platform, filename in (("win32", "GgmlOps.dll"), ("darwin", "libGgmlOps.dylib"), ("linux", "libGgmlOps.so")):
            with self.subTest(platform=platform):
                self.assertEqual(Path("bin") / filename, VISION.native_library_path(Path("bin"), platform))

    def test_default_fixture_contract_remains_unchanged(self):
        for name, original in WORKFLOWS.CASES.items():
            self.assertEqual(original, WORKFLOWS.case_spec(name, "c1-i0"))

    def test_windows_shell_commands_and_concurrent_oracles_are_distinct(self):
        for shell, command in (("powershell", "Write-Output"), ("cmd", "echo")):
            cases = [WORKFLOWS.case_spec("shell_run", f"c4-i{i}", True, shell) for i in range(4)]
            self.assertEqual(4, len({case["expected"] for case in cases}))
            for case in cases:
                self.assertIn(command, case["prompt"])
                self.assertIn(case["expected"], case["prompt"])
                self.assertNotIn("printf", case["prompt"])
                self.assertIn("Windows", case["prompt"])

    def test_windows_code_prompts_keep_independent_function_contract(self):
        for name in ("code_generation_run", "code_edit_run"):
            original = WORKFLOWS.CASES[name]
            case = WORKFLOWS.case_spec(name, "c1-i0", True, "powershell")
            self.assertEqual(original["tools"], case["tools"])
            for key, value in original["artifact"].items():
                self.assertEqual(value, case["artifact"][key])
            self.assertIn(case["artifact"]["probe"], case["prompt"])
            self.assertIn("do not use POSIX shell heredocs", case["prompt"])
        with self.assertRaises(ValueError):
            WORKFLOWS.case_spec("shell_run", "trial", target_shell="guess")

    def test_missing_sandbox_does_not_silently_become_unconfined(self):
        for platform in ("win32", "darwin"):
            with self.assertRaisesRegex(RuntimeError, "confined mode requires Linux"):
                VERIFIER.execution_command(Path("stage"), "bwrap", False, "python", platform)
        command = VERIFIER.execution_command(Path("stage"), "unused", True, sys.executable, "win32")
        self.assertEqual([sys.executable, "-I", "verify.py"], command)


class ArtifactOracle(unittest.TestCase):
    def run_verifier(self, source, scenario="code_generation_run", status="ok", url=None, earlier_source=None, diagnose=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "retained artifacts"
            store.mkdir()
            filename = VERIFIER.CONTRACTS[scenario][0]
            (store / filename).write_text(source, encoding="utf-8")
            (root / filename).write_text(source, encoding="utf-8")
            files = []
            if earlier_source is not None:
                earlier = store / "earlier"
                earlier.mkdir()
                (earlier / filename).write_text(earlier_source, encoding="utf-8")
                files.append({"url": f"/api/code/artifacts/earlier/{filename}"})
            files.append({"url": url or f"/api/code/artifacts/{filename}"})
            workflow = root / "workflow.json"
            workflow.write_text(json.dumps({"cases": [{"scenario": scenario, "trial": "cpu-fixture",
                "status": status, "events": [{"files": files}]}]}), encoding="utf-8-sig")
            output = root / "results.json"
            result = subprocess.run([sys.executable, str(DIRECTORY / "verify-agent-code-artifacts.py"),
                "--workflow-report", str(workflow), "--artifact-store", str(store), "--output", str(output),
                "--sandbox-off"] + (["--diagnose-incomplete"] if diagnose else []), capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertTrue(output.exists(), result.stdout + result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("unconfined", report["execution_mode"])
            self.assertTrue(report["run_complete"])
            return result.returncode, report["cases"][0]

    def test_correct_generated_functions_execute_and_pass_independent_inputs(self):
        for scenario, code in (("code_generation_run", "def sum_numbers(n): return n * (n + 1) // 2\n"),
                               ("code_edit_run", "def parity(n): return 'even' if n % 2 == 0 else 'odd'\n")):
            with self.subTest(scenario=scenario):
                exit_code, case = self.run_verifier(code, scenario)
                self.assertEqual(0, exit_code)
                self.assertEqual("ok", case["status"])
                self.assertIn("TENSORSHARP_INDEPENDENT_CODE_CHECK=", case["stdout"])

    def test_wrong_function_or_float_type_does_not_pass_a_successful_workflow_claim(self):
        for code in ("def sum_numbers(n): return 703\n", "def sum_numbers(n): return n * (n + 1) / 2\n"):
            exit_code, case = self.run_verifier(code)
            self.assertEqual(1, exit_code)
            self.assertEqual("fail", case["status"])

    def test_latest_source_version_wins_over_an_earlier_correct_draft(self):
        exit_code, case = self.run_verifier("def sum_numbers(n): return 703\n",
            earlier_source="def sum_numbers(n): return n * (n + 1) // 2\n")
        self.assertEqual(1, exit_code)
        self.assertEqual("/api/code/artifacts/sum_numbers.py", case["artifact_url"])

    def test_failed_original_workflow_and_escaping_route_are_rejected(self):
        correct = "def sum_numbers(n): return n * (n + 1) // 2\n"
        exit_code, case = self.run_verifier(correct, status="blocked")
        self.assertEqual(1, exit_code)
        self.assertIn("Original workflow did not pass", case["detail"])
        exit_code, case = self.run_verifier(correct, url="/api/code/artifacts/%2e%2e/sum_numbers.py")
        self.assertEqual(1, exit_code)
        self.assertIn("escapes artifact store", case["detail"])
        self.assertNotIn("exit_code", case)

    def test_incomplete_diagnostic_can_check_code_without_passing_failed_workflow(self):
        exit_code, case = self.run_verifier("def sum_numbers(n): return n * (n + 1) // 2\n",
                                            status="fail", diagnose=True)
        self.assertEqual(1, exit_code)
        self.assertEqual("fail", case["status"])
        self.assertTrue(case["function_checks_passed"])
        self.assertEqual("fail", case["original_workflow_status"])


if __name__ == "__main__":
    unittest.main()
