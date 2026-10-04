"""DMG finalization control flow with no signing tools or Apple calls.

Mocks cannot prove signatures, ticket validity, Gatekeeper, or real notarization.
"""
import importlib.util
from contextlib import redirect_stdout
import io
import fcntl
import json
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location(
    "dmg_finalizer", Path(__file__).resolve().parents[1] / "finalize-tensoragent-dmg.py")
finisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finisher)
APP_ID = "11111111-2222-4333-8444-555555555555"
DMG_ID = "66666666-7777-4888-8999-000000000000"


class Tools:
    def __init__(self, root):
        self.root, self.calls, self.audits = root, [], []
        self.app_status = self.dmg_status = "Accepted"
        self.fail_open_once = self.fail_mounted_once = False
        self.fail_app_gatekeeper_once = False
        self.submit_receipt = True
        self.wait_returncode = None
        self.wait_override_id = None
        self.replace_mounted_payload = False
        self.dmg_cdhash = "a" * 40
        self.wait_hook = None
        self.malformed_attach = False
        self.log_override_id = None

    def audit(self, app, run):
        self.audits.append(Path(app))
        return "ABC1234567"

    def run(self, args, **kwargs):
        self.calls.append(args)
        stdout, stderr, code = "", "", 0
        name = args[0]
        if args[:2] == ["xcrun", "vtool"]:
            stdout = "minos 14.0"
        elif name == "otool":
            stdout = "library:\n\t@rpath/libGgmlOps.dylib (compatibility version 0.0.0)\n\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0)\n"
        elif name == "ditto":
            shutil.copytree(args[1], args[2], symlinks=True, dirs_exist_ok=True)
        elif args[:3] == ["xcrun", "notarytool", "wait"]:
            if self.wait_hook:
                self.wait_hook(args[3])
            status = self.app_status if args[3] == APP_ID else self.dmg_status
            stdout = json.dumps({"id": self.wait_override_id or args[3], "status": status})
            code = 75 if status == "In Progress" else 0
            if self.wait_returncode is not None:
                code = self.wait_returncode
        elif args[:3] == ["xcrun", "notarytool", "submit"]:
            stdout = json.dumps({"id": DMG_ID, "status": "In Progress"}) if self.submit_receipt else ""
        elif args[:3] == ["xcrun", "notarytool", "log"]:
            status = self.app_status if args[3] == APP_ID else self.dmg_status
            Path(args[-1]).write_text(json.dumps({
                "jobId": self.log_override_id or args[3], "status": status,
                "statusCode": 0 if status == "Accepted" else 4000,
                "issues": [{"severity": "warning", "message": "mock warning"}]}))
        elif args[:3] == ["xcrun", "stapler", "staple"]:
            path = Path(args[-1])
            if path.is_dir():
                (path / "Contents/mock-ticket").write_text("stapled app")
            else:
                with path.open("ab") as stream:
                    stream.write(b"stapled dmg")
        elif args[:3] == ["xcrun", "stapler", "validate"]:
            path = Path(args[-1])
            if path.suffix == ".dmg" and b"stapled dmg" not in path.read_bytes():
                code = 1
        elif name == "codesign":
            if "--sign" in args:
                with Path(args[-1]).open("ab") as stream:
                    stream.write(b"signed")
            elif "--display" in args:
                stderr = "Authority=Developer ID Application: TensorSharp (ABC1234567)\nTeamIdentifier=ABC1234567\nTimestamp=Oct 4, 2026 at 12:00:00 PM\nCDHash=" + self.dmg_cdhash
        elif args[:2] == ["hdiutil", "create"]:
            disk = Path(args[args.index("-srcfolder") + 1])
            shutil.copytree(disk, self.root / "snapshot", symlinks=True, dirs_exist_ok=True)
            Path(args[-1]).write_bytes(b"mock UDZO")
        elif args[:2] == ["hdiutil", "attach"]:
            mount = Path(args[args.index("-mountpoint") + 1])
            shutil.copytree(self.root / "snapshot", mount, symlinks=True, dirs_exist_ok=True)
            if self.replace_mounted_payload:
                (mount / "TensorAgent.app/Contents/MacOS/TensorAgent.Maui").write_bytes(b"another signed app")
            stdout = "malformed plist" if self.malformed_attach else plistlib.dumps({"system-entities": [{"mount-point": str(mount)}]}).decode()
        elif args[:2] == ["hdiutil", "detach"]:
            shutil.rmtree(args[2])
        elif name == "spctl":
            if "open" in args and self.fail_open_once:
                self.fail_open_once, code = False, 1
            if "execute" in args and Path(args[-1]).parent.name == "mount" and self.fail_mounted_once:
                self.fail_mounted_once, code = False, 1
            elif "execute" in args and self.fail_app_gatekeeper_once:
                self.fail_app_gatekeeper_once, code = False, 1
        return subprocess.CompletedProcess(args, code, stdout, stderr)

    def count(self, prefix):
        return sum(command[:len(prefix)] == prefix for command in self.calls)


class DmgFinalizerTests(unittest.TestCase):
    def setUp(self):
        redirect = redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def fixture(self, root):
        app = root / "TensorAgent.app"
        main = app / "Contents/MacOS/TensorAgent.Maui"
        main.parent.mkdir(parents=True)
        main.write_bytes(b"mock executable")
        main.chmod(0o755)
        native = app / "Contents/MonoBundle/libGgmlOps.dylib"
        native.parent.mkdir()
        native.write_bytes(b"mock native binary")
        (app / "Contents/Resources/webui").mkdir(parents=True)
        (app / "Contents/Resources/webui/index.html").write_text("mock UI")
        (app / "Contents/Resources/skills").mkdir()
        (app / "Contents/Info.plist").write_bytes(plistlib.dumps({
            "CFBundleExecutable": main.name, "CFBundleShortVersionString": "2026.10.3",
            "LSMinimumSystemVersion": "14.0"}))
        options = SimpleNamespace(app=str(app), version="2026.10.03", identity="A" * 40,
                                  notary_profile="mock-profile", app_submission_id=APP_ID,
                                  output=str(root / "artifacts"), timeout="48h")
        return options, Tools(root)

    def finalizer(self, options, tools):
        return finisher.Finalizer(options, run=tools.run, audit=tools.audit)

    def test_acceptance_requires_accepted_status_and_successful_wait(self):
        finisher.accepted({"status": "Accepted"}, 0)
        for status, code in (("Invalid", 0), ("In Progress", 0), ("Accepted", 75)):
            with self.subTest(status=status, code=code), self.assertRaises(ValueError):
                finisher.accepted({"status": status}, code)

    def test_wait_does_not_cache_accepted_output_from_a_failed_command(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            tools.wait_returncode = 75
            job = self.finalizer(options, tools)
            with self.assertRaises(ValueError):
                job.wait(APP_ID, "app")
            tools.wait_returncode = 0
            job.wait(APP_ID, "app")
            self.assertEqual(tools.count(["xcrun", "notarytool", "wait"]), 2)

    def test_wait_rejects_receipt_for_a_different_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            tools.wait_override_id = DMG_ID
            with self.assertRaisesRegex(ValueError, "different submission"):
                self.finalizer(options, tools).wait(APP_ID, "app")
            self.assertEqual(tools.count(["xcrun", "notarytool", "log"]), 0)

    def test_successful_accepted_wait_and_matching_log_resume_without_profile_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            job = self.finalizer(options, tools)
            job.wait(APP_ID, "app")
            job.wait(APP_ID, "app")
            self.assertEqual(tools.count(["xcrun", "notarytool", "wait"]), 1)
            self.assertEqual(tools.count(["xcrun", "notarytool", "log"]), 1)

    def test_mismatched_cached_log_is_refetched_instead_of_trusted(self):
        for changes in ({"jobId": DMG_ID}, {"status": "Invalid"}, {"statusCode": 4000}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                options, tools = self.fixture(Path(directory))
                job = self.finalizer(options, tools)
                job.wait(APP_ID, "app")
                log = job.evidence / "app-log.json"
                document = finisher.read_json(log)
                document.update(changes)
                finisher.write_json(log, document)
                job.wait(APP_ID, "app")
                self.assertEqual(tools.count(["xcrun", "notarytool", "wait"]), 1)
                self.assertEqual(tools.count(["xcrun", "notarytool", "log"]), 2)
                self.assertEqual(finisher.read_json(log)["jobId"], APP_ID)

    def test_mismatched_downloaded_accepted_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            tools.log_override_id = DMG_ID
            with self.assertRaisesRegex(ValueError, "log does not match"):
                self.finalizer(options, tools).wait(APP_ID, "app")

    def test_invalid_app_receipt_stops_before_dmg_creation_and_preserves_apple_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            tools.app_status = "Invalid"
            job = self.finalizer(options, tools)
            with self.assertRaisesRegex(ValueError, "Invalid"):
                job.execute()
            self.assertEqual(tools.count(["hdiutil", "create"]), 0)
            self.assertEqual(tools.count(["xcrun", "notarytool", "submit"]), 0)
            self.assertFalse(job.final.exists())
            self.assertEqual(finisher.read_json(job.evidence / "app-warnings.json")[0]["severity"], "warning")

    def test_timeout_resumes_existing_dmg_receipt_without_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            tools.dmg_status = "In Progress"
            job = self.finalizer(options, tools)
            # Construct another worker before the receipt is saved. Its execute
            # must reload under the lock rather than use this stale empty state.
            resumed = self.finalizer(options, tools)
            with self.assertRaisesRegex(ValueError, "In Progress"):
                job.execute()
            self.assertEqual(job.state["dmg_submission_id"], DMG_ID)
            self.assertFalse(job.final.exists())
            self.assertFalse((job.evidence / "dmg-log.json").exists())
            tools.dmg_status = "Accepted"
            result = resumed.execute()
            self.assertEqual(tools.count(["xcrun", "notarytool", "submit"]), 1)
            self.assertEqual(tools.count(["hdiutil", "create"]), 1)
            self.assertTrue(result.exists())
            self.assertFalse((Path(options.app) / "Contents/mock-ticket").exists())
            self.assertIn(job.stage / "mount/TensorAgent.app", tools.audits)
            self.assertEqual(tools.count(["hdiutil", "detach"]), 1)
            checksum = Path(options.output) / ("SHA256SUMS-" + resumed.stem + ".txt")
            self.assertEqual(checksum.read_text(), finisher.sha256(result) + "  " + result.name + "\n")

    def test_invalid_dmg_never_promotes_and_rerun_reuses_the_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            tools.dmg_status = "Invalid"
            for _ in range(2):
                job = self.finalizer(options, tools)
                with self.assertRaisesRegex(ValueError, "Invalid"):
                    job.execute()
                self.assertFalse(job.final.exists())
            self.assertEqual(tools.count(["xcrun", "notarytool", "submit"]), 1)

    def test_changed_submitted_dmg_cannot_reuse_pending_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            tools.dmg_status = "In Progress"
            job = self.finalizer(options, tools)
            with self.assertRaises(ValueError):
                job.execute()
            with job.dmg.open("ab") as stream:
                stream.write(b"unexpected modification")
            with self.assertRaisesRegex(ValueError, "Submitted staged DMG changed"):
                self.finalizer(options, tools).execute()
            self.assertEqual(tools.count(["xcrun", "notarytool", "submit"]), 1)
            self.assertFalse(job.final.exists())

    def test_post_staple_gate_failure_resumes_with_changed_digest_and_preserves_existing_release(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            tools.fail_open_once = True
            job = self.finalizer(options, tools)
            job.final.write_bytes(b"previous public artifact")
            with self.assertRaisesRegex(RuntimeError, "dmg-gatekeeper"):
                job.execute()
            self.assertEqual(job.final.read_bytes(), b"previous public artifact")
            self.assertNotEqual(job.state["dmg_signed"], job.state["dmg_stapled"])
            resumed = self.finalizer(options, tools)
            resumed.execute()
            self.assertEqual(tools.count(["xcrun", "notarytool", "submit"]), 1)
            self.assertNotEqual(job.final.read_bytes(), b"previous public artifact")

    def test_mounted_app_failure_detaches_and_never_promotes(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            tools.fail_mounted_once = True
            job = self.finalizer(options, tools)
            with self.assertRaisesRegex(RuntimeError, "mounted-app-gatekeeper"):
                job.execute()
            self.assertEqual(tools.count(["hdiutil", "detach"]), 1)
            self.assertFalse(job.final.exists())
            self.assertFalse((job.stage / "mount").exists())

    def test_malformed_mount_response_still_detaches_known_private_mountpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            tools.malformed_attach = True
            job = self.finalizer(options, tools)
            with self.assertRaises(plistlib.InvalidFileException):
                job.execute()
            self.assertEqual(tools.count(["hdiutil", "detach"]), 1)
            self.assertFalse(job.final.exists())
            self.assertFalse((job.stage / "mount").exists())

    def test_source_change_during_app_wait_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))

            def change_source(submission_id):
                if submission_id == APP_ID:
                    (Path(options.app) / "Contents/MacOS/TensorAgent.Maui").write_bytes(b"changed during wait")

            tools.wait_hook = change_source
            with self.assertRaisesRegex(ValueError, "source app changed"):
                self.finalizer(options, tools).execute()
            self.assertEqual(tools.count(["ditto"]), 0)

    def test_dmg_changed_during_wait_requires_an_independently_valid_existing_ticket(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            job = self.finalizer(options, tools)

            def change_dmg(submission_id):
                if submission_id == DMG_ID:
                    with job.dmg.open("ab") as stream:
                        stream.write(b"unexpected modification during wait")

            tools.wait_hook = change_dmg
            with self.assertRaisesRegex(RuntimeError, "recover-dmg-ticket"):
                job.execute()
            self.assertFalse(job.final.exists())

    def test_exclusive_stage_lock_prevents_concurrent_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            job = self.finalizer(options, tools)
            with (job.stage / "lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, "Another finalizer"):
                    job.execute()
            self.assertFalse(tools.calls)

    def test_mounted_payload_replacement_is_rejected_despite_mock_signature_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            tools.replace_mounted_payload = True
            job = self.finalizer(options, tools)
            with self.assertRaisesRegex(ValueError, "Mounted app differs"):
                job.execute()
            self.assertEqual(tools.count(["hdiutil", "detach"]), 1)
            self.assertFalse(job.final.exists())

    def test_replaced_staged_app_cannot_become_the_expected_payload_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            tools.fail_app_gatekeeper_once = True
            job = self.finalizer(options, tools)
            with self.assertRaisesRegex(RuntimeError, "app-gatekeeper"):
                job.execute()
            self.assertTrue(job.state["app_copied"])
            self.assertNotIn("dmg_signed", job.state)
            (job.staged_app / "Contents/MacOS/TensorAgent.Maui").write_bytes(b"another valid signed app")
            with self.assertRaisesRegex(ValueError, "Staged main binary differs"):
                self.finalizer(options, tools).execute()
            self.assertEqual(tools.count(["hdiutil", "create"]), 0)
            self.assertEqual(tools.count(["xcrun", "notarytool", "submit"]), 0)
            self.assertFalse(job.final.exists())

    def test_crash_after_staple_recovers_only_with_same_code_identity_and_valid_ticket(self):
        class Interrupted(finisher.Finalizer):
            def staple(self, path, label):
                super().staple(path, label)
                if label == "dmg":
                    raise RuntimeError("crash before post-staple checkpoint")

        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            job = Interrupted(options, run=tools.run, audit=tools.audit)
            with self.assertRaisesRegex(RuntimeError, "crash before"):
                job.execute()
            self.assertNotIn("dmg_stapled", job.state)
            self.assertTrue(job.state["dmg_accepted"])
            self.assertNotEqual(finisher.sha256(job.dmg), job.state["dmg_signed"])
            self.finalizer(options, tools).execute()
            self.assertTrue(job.final.exists())
            self.assertEqual(tools.count(["xcrun", "notarytool", "submit"]), 1)

    def test_accepted_receipt_does_not_allow_replacement_signed_payload_in_crash_window(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            tools.fail_open_once = True
            job = self.finalizer(options, tools)
            with self.assertRaises(RuntimeError):
                job.execute()
            state = finisher.read_json(job.checkpoint)
            state.pop("dmg_stapled")
            finisher.write_json(job.checkpoint, state)
            tools.dmg_cdhash = "b" * 40
            with self.assertRaisesRegex(ValueError, "signed payload differs"):
                self.finalizer(options, tools).execute()
            self.assertFalse(job.final.exists())
            self.assertEqual(tools.count(["xcrun", "notarytool", "submit"]), 1)

    def test_ambiguous_submit_without_receipt_fails_closed_instead_of_resubmitting(self):
        with tempfile.TemporaryDirectory() as directory:
            options, tools = self.fixture(Path(directory))
            tools.submit_receipt = False
            with self.assertRaises(ValueError):
                self.finalizer(options, tools).execute()
            with self.assertRaisesRegex(ValueError, "may have reached Apple"):
                self.finalizer(options, tools).execute()
            self.assertEqual(tools.count(["xcrun", "notarytool", "submit"]), 1)


if __name__ == "__main__":
    unittest.main()
