"""Exercise wrapper argument forwarding using a fake npx; no browser or downloads."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


WRAPPER = Path(__file__).resolve().parents[3] / "TensorAgent/skills/playwright/scripts/playwright_cli.sh"


def find_bash():
    """Choose a shell that can read this checkout, avoiding Windows' WSL shim."""
    candidates = []
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            git_root = Path(git).resolve().parent.parent
            candidates.extend((git_root / "bin/bash.exe", git_root / "usr/bin/bash.exe"))
        for variable, suffix in (("ProgramFiles", "Git"), ("ProgramFiles(x86)", "Git"),
                                 ("LOCALAPPDATA", "Programs/Git")):
            if os.environ.get(variable):
                git_root = Path(os.environ[variable]) / suffix
                candidates.extend((git_root / "bin/bash.exe", git_root / "usr/bin/bash.exe"))
        candidates.extend(Path(entry.strip('"')) / "bash.exe"
                          for entry in os.environ.get("PATH", "").split(os.pathsep) if entry)
    else:
        bash = shutil.which("bash")
        if bash:
            candidates.append(Path(bash))
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        try:
            result = subprocess.run(
                [str(candidate), "--noprofile", "--norc", "-c", 'test -f "$1"',
                 "wrapper-access-check", WRAPPER.as_posix()],
                capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return candidate
    raise RuntimeError("A Bash able to access the local checkout is required; "
                       "on Windows install Git Bash (the WSL bash.exe shim is unsuitable).")


class PlaywrightWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = find_bash()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="playwright wrapper ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        fixture = self.bin / "npx_fixture.py"
        fixture.write_bytes(b"import json,os,sys\nprint(json.dumps(sys.argv[1:]))\n"
                            b"sys.exit(int(os.environ.get('FAKE_NPX_EXIT', '0')))\n")
        npx = self.bin / "npx"
        # A Python shebang with a Windows executable path cannot be run by Git
        # Bash. Use an LF shell shim and quoted paths so spaces survive too.
        npx.write_bytes(b'#!/usr/bin/env bash\n'
                        b'exec "$WRAPPER_TEST_PYTHON" "$WRAPPER_TEST_NPX" "$@"\n')
        npx.chmod(0o755)
        self.env = os.environ.copy()
        self.env["PATH"] = str(self.bin) + os.pathsep + self.env.get("PATH", "")
        self.env["WRAPPER_TEST_PYTHON"] = Path(sys.executable).as_posix()
        self.env["WRAPPER_TEST_NPX"] = fixture.as_posix()
        if os.name == "nt":
            # Git Bash otherwise rewrites POSIX-looking CLI arguments when it
            # crosses into native Python, hiding argument-forwarding failures.
            self.env["MSYS2_ARG_CONV_EXCL"] = "*"
        self.env.pop("PLAYWRIGHT_CLI_SESSION", None)
        self.env.pop("FAKE_NPX_EXIT", None)

    def run_wrapper(self, *args, expected_exit=0):
        result = subprocess.run([str(self.bash), "--noprofile", "--norc", WRAPPER.as_posix(), *args],
                                cwd=self.root, env=self.env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(expected_exit, result.returncode,
                         f"wrapper stdout: {result.stdout!r}; stderr: {result.stderr!r}")
        self.assertEqual("", result.stderr)
        argv = json.loads(result.stdout)
        self.assertEqual(["--yes", "--prefer-offline", "--package", "@playwright/cli@0.1.21", "playwright-cli"], argv[:5])
        return argv[5:]

    def test_subprocess_preserves_literal_arguments_and_propagates_exit(self):
        supplied = ["fill", "e3", '你好 & echo bad | "quotes" %PATH% $HOME `literal` ; (value) \\',
                    "/tmp/literal/path", "C:\\literal path\\file"]
        self.env["FAKE_NPX_EXIT"] = "23"
        self.assertEqual(supplied, self.run_wrapper(*supplied, expected_exit=23))
        self.assertFalse((self.root / "output").exists())

    def generated_file(self, argv):
        names = [arg.split("=", 1)[1] for arg in argv if arg.startswith("--filename=")]
        self.assertEqual(1, len(names))
        path = Path(names[0])
        self.assertEqual(Path("output/playwright"), path.parent)
        self.assertTrue((self.root / path).is_file())
        return path

    def test_bare_snapshots_get_distinct_reserved_files(self):
        first = self.run_wrapper("snapshot")
        second = self.run_wrapper("snapshot")
        self.assertEqual("snapshot", first[-1])
        self.assertNotEqual(self.generated_file(first), self.generated_file(second))

    def test_target_and_snapshot_options_keep_boundaries(self):
        supplied = ["--json", "--session", "named session", "snapshot", "main > article",
                    "--depth", "3", "--boxes"]
        actual = self.run_wrapper(*supplied)
        self.generated_file(actual)
        self.assertEqual(supplied, actual[1:])

    def test_snapshot_flags_can_precede_command(self):
        for supplied in (["--depth", "3", "--boxes", "snapshot"],
                         ["--raw", "--session=snapshot", "snapshot"],
                         ["--raw", "false", "--json", "true", "snapshot"],
                         ["--boxes", "true", "--help", "false", "snapshot"],
                         ["-s", "snapshot", "snapshot"]):
            with self.subTest(args=supplied):
                actual = self.run_wrapper(*supplied)
                self.generated_file(actual)
                self.assertEqual(supplied, actual[1:])

    def test_explicit_filename_forms_preserved_without_generated_artifacts(self):
        for supplied in (["snapshot", "--filename", "chosen name.yml"],
                         ["snapshot", "--filename=chosen name.yml"],
                         ["snapshot", "--filename="],
                         ["--filename", "chosen.yml", "snapshot"]):
            with self.subTest(args=supplied):
                self.assertEqual(supplied, self.run_wrapper(*supplied))
        self.assertFalse((self.root / "output").exists())

    def test_help_and_version_do_not_create_files(self):
        for supplied in (["snapshot", "--help"], ["--help", "snapshot"],
                         ["snapshot", "-h"], ["--version", "snapshot"],
                         ["--help=true", "snapshot"], ["--version", "true", "snapshot"]):
            with self.subTest(args=supplied):
                self.assertEqual(supplied, self.run_wrapper(*supplied))
        self.assertFalse((self.root / "output").exists())

    def test_snapshot_as_other_command_argument_or_session_name_is_untouched(self):
        for supplied in (["fill", "e3", "snapshot"], ["open", "snapshot"],
                         ["--session", "snapshot", "goto", "https://example.test"],
                         ["-s=snapshot", "close"]):
            with self.subTest(args=supplied):
                self.assertEqual(supplied, self.run_wrapper(*supplied))
        self.assertFalse((self.root / "output").exists())

    def test_end_of_options_keeps_generated_filename_as_option(self):
        supplied = ["snapshot", "--", "--filename=literal-target"]
        actual = self.run_wrapper(*supplied)
        self.assertTrue(actual[0].startswith("--filename=output/playwright/"))
        self.assertEqual(supplied, actual[1:])

    def test_environment_session_is_applied_only_without_explicit_session(self):
        self.env["PLAYWRIGHT_CLI_SESSION"] = "from environment"
        actual = self.run_wrapper("snapshot")
        self.assertEqual(["--session", "from environment"], actual[:2])
        self.generated_file(actual)
        for flag in (["--session", "supplied"], ["--session=supplied"],
                     ["-s", "supplied"], ["-s=supplied"]):
            with self.subTest(flag=flag):
                supplied = flag + ["snapshot"]
                actual = self.run_wrapper(*supplied)
                self.generated_file(actual)
                self.assertEqual(supplied, actual[1:])


if __name__ == "__main__":
    unittest.main()
