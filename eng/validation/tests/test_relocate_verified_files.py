"""Small temporary-file tests only. No models, user directories, GPU or builds.

The retirement tests exercise actual unlink/rmdir and a real Windows junction
(POSIX directory symlink elsewhere), always inside a fresh TemporaryDirectory.
"""
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


TOOL = Path(__file__).resolve().parents[1] / "relocate-verified-files.py"
SPEC = importlib.util.spec_from_file_location("relocate_verified_files", TOOL)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class RelocationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ts-relocate-test-")
        self.root = Path(self.temporary.name).resolve()
        self.source_root = self.root / "sources"
        self.destination_root = self.root / "destinations"
        self.receipt_root = self.root / "receipts"
        for root in (self.source_root, self.destination_root, self.receipt_root):
            root.mkdir()
        self.source = self.source_root / "scope with spaces 中文"
        self.source.mkdir()
        self.destination = self.destination_root / "archive with spaces 中文"
        self.receipt = self.receipt_root / "operation"
        self.add("first.bin", b"first payload")
        self.add("nested/second.bin", b"second payload with tail")
        (self.source / "empty").mkdir()
        self.config = M.Config(self.source_root, self.source, self.destination_root, self.destination,
                               self.receipt_root, self.receipt, minimum_free_bytes=0,
                               chunk_bytes=7, progress_bytes=13)

    def tearDown(self):
        # Explicitly detach the test alias before TemporaryDirectory cleanup.
        if os.path.lexists(self.source) and M.reparse(self.source.lstat()):
            if os.name == "nt" and not self.source.is_symlink():
                self.source.rmdir()
            else:
                self.source.unlink()
        self.temporary.cleanup()

    def add(self, relative, content):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def run_tool(self, config=None, *, resume=False, reporter=None):
        return M.Relocator(config or self.config, resume=resume, reporter=reporter).execute()

    def stored(self):
        return json.loads((self.receipt / "receipt.json").read_text(encoding="utf-8"))

    def test_copy_only_full_hash_snapshots_progress_empty_dirs_and_resume(self):
        progress = []
        receipt = self.run_tool(reporter=progress.append)
        self.assertEqual("copy_complete", receipt["status"])
        self.assertEqual(2, len(receipt["files"]))
        self.assertTrue((self.source / "first.bin").exists())
        self.assertTrue((self.destination / "empty").is_dir())
        self.assertEqual(0, receipt["logical_retired_bytes"])
        self.assertGreater(len(progress), 5)
        self.assertIn("free", receipt["disk_samples"][0]["volumes"]["source"])
        for relative, item in receipt["files"].items():
            data = (self.source / relative).read_bytes()
            self.assertEqual(data, (self.destination / relative).read_bytes())
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
            self.assertEqual("verified", item["state"])
            self.assertEqual(item["source_snapshot"]["size"], item["destination_snapshot"]["size"])
            self.assertEqual(item["source_snapshot"]["mtime_ns"], item["destination_snapshot"]["mtime_ns"])
        self.assertEqual("copy_complete", self.run_tool(resume=True)["status"])

    def test_same_size_different_content_never_overwritten_or_retired(self):
        self.destination.mkdir()
        target = self.destination / "first.bin"
        target.write_bytes(b"X" * len((self.source / "first.bin").read_bytes()))
        config = dataclasses.replace(self.config, retire_source=True, junction=True)
        with self.assertRaisesRegex(M.Refusal, "Conflicting existing destination content"):
            self.run_tool(config)
        self.assertEqual(b"X" * 13, target.read_bytes())
        self.assertEqual(b"first payload", (self.source / "first.bin").read_bytes())
        self.assertEqual([], self.stored()["deletion_intents"])

    def test_existing_equal_content_reused_only_after_full_hash(self):
        self.destination.mkdir()
        target = self.destination / "first.bin"
        target.write_bytes(b"first payload")
        os.utime(target, ns=(1_000_000_000, 1_000_000_000))
        old = M.signature(target)
        result = self.run_tool()
        self.assertEqual(old, result["files"]["first.bin"]["destination_snapshot"])
        self.assertEqual("existing_full_hash_match", result["files"]["first.bin"]["disposition"])

    def test_mutation_after_copy_before_retirement_blocks_entire_scope(self):
        original = M.Relocator.retirement_preflight
        def mutate_then_preflight(operation):
            path = self.source / "nested/second.bin"
            path.write_bytes(b"changed after copy")
            return original(operation)
        with patch.object(M.Relocator, "retirement_preflight", mutate_then_preflight):
            with self.assertRaisesRegex(M.Refusal, "Source changed"):
                self.run_tool(dataclasses.replace(self.config, retire_source=True, junction=True))
        self.assertTrue((self.source / "first.bin").exists())
        self.assertTrue((self.source / "nested/second.bin").exists())
        self.assertEqual([], self.stored()["deletion_intents"])

    def test_same_length_mutation_even_restored_mtime_detected_before_retirement(self):
        original = M.Relocator.retirement_preflight
        def mutate_then_preflight(operation):
            path = self.source / "first.bin"
            snapshot = path.stat()
            path.write_bytes(b"Z" * snapshot.st_size)
            os.utime(path, ns=(snapshot.st_atime_ns, snapshot.st_mtime_ns))
            return original(operation)
        with patch.object(M.Relocator, "retirement_preflight", mutate_then_preflight):
            with self.assertRaises(M.Refusal):
                self.run_tool(dataclasses.replace(self.config, retire_source=True))
        self.assertTrue((self.source / "nested/second.bin").exists())
        self.assertEqual([], self.stored()["deletion_intents"])

    def test_resume_rejects_source_mutation_and_preserves_verified_destination(self):
        self.run_tool()
        self.add("first.bin", b"source mutated")
        with self.assertRaisesRegex(M.Refusal, "Source changed"):
            self.run_tool(resume=True)
        self.assertEqual(b"first payload", (self.destination / "first.bin").read_bytes())

    def test_interrupted_part_is_recorded_and_restarted_not_published_without_hash(self):
        with patch.object(M, "publish_no_replace", side_effect=OSError("injected prepublication interruption")):
            with self.assertRaisesRegex(OSError, "injected"):
                self.run_tool()
        failed = self.stored()
        item = failed["files"]["first.bin"]
        self.assertEqual("publishing", item["state"])
        part = Path(item["part"])
        self.assertTrue(part.exists())
        self.assertFalse((self.destination / "first.bin").exists())
        part.write_bytes(b"corrupt incomplete owned partial")
        resumed = self.run_tool(resume=True)
        self.assertEqual("copy_complete", resumed["status"])
        self.assertFalse(part.exists())
        self.assertEqual(b"first payload", (self.destination / "first.bin").read_bytes())

    def test_partial_copy_midstream_mutation_never_publishes_or_retires(self):
        mutated = False
        def observe(progress):
            nonlocal mutated
            if progress["phase"] == "copy" and not mutated:
                mutated = True
                self.add("first.bin", b"mutation during the copy")
        with self.assertRaises(M.Refusal):
            self.run_tool(dataclasses.replace(self.config, retire_source=True), reporter=observe)
        self.assertFalse((self.destination / "first.bin").exists())
        self.assertTrue((self.source / "nested/second.bin").exists())

    def test_atomic_publication_refuses_destination_race(self):
        original = M.publish_no_replace
        def race(part, target):
            target.write_bytes(b"external unrelated data")
            return original(part, target)
        with patch.object(M, "publish_no_replace", race):
            with self.assertRaises(FileExistsError):
                self.run_tool(dataclasses.replace(self.config, retire_source=True))
        self.assertEqual(b"external unrelated data", (self.destination / "first.bin").read_bytes())
        self.assertTrue((self.source / "first.bin").exists())

    def test_named_root_boundaries_overlap_and_partial_retirement_refuse(self):
        invalid = [dataclasses.replace(self.config, source=self.source_root),
                   dataclasses.replace(self.config, source=self.root / "outside"),
                   dataclasses.replace(self.config, destination=self.source / "nested"),
                   dataclasses.replace(self.config, include_files=("../outside",)),
                   dataclasses.replace(self.config, include_files=("first.bin",), retire_source=True),
                   dataclasses.replace(self.config, junction=True)]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(M.Refusal):
                M.Relocator(config)
        self.assertFalse(self.receipt.exists())

    def test_include_exact_file_only_no_other_payload(self):
        result = self.run_tool(dataclasses.replace(self.config, include_files=("nested/second.bin",)))
        self.assertEqual(["nested/second.bin"], list(result["files"]))
        self.assertFalse((self.destination / "first.bin").exists())
        self.assertTrue((self.source / "first.bin").exists())

    def make_directory_alias(self, link, target):
        if os.name == "nt":
            ps = shutil.which("pwsh") or shutil.which("powershell")
            if not ps:
                self.skipTest("PowerShell unavailable for tiny Windows junction fixture")
            script = self.root / "fixture-junction.ps1"
            script.write_text("param([string]$Link,[string]$Target)\n$ErrorActionPreference='Stop'\n"
                              "New-Item -ItemType Junction -Path $Link -Target $Target | Out-Null\n", encoding="utf-8")
            import subprocess
            subprocess.run([ps, "-NoProfile", "-NonInteractive", "-File", str(script),
                            "-Link", str(link), "-Target", str(target)], check=True, capture_output=True)
        else:
            link.symlink_to(target, target_is_directory=True)

    def remove_directory_alias(self, link):
        if os.name == "nt" and not link.is_symlink():
            link.rmdir()
        else:
            link.unlink()

    def test_source_and_destination_reparse_trees_refuse(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "untouched.bin").write_bytes(b"outside")
        link = self.source / "alias"
        self.make_directory_alias(link, outside)
        try:
            with self.assertRaisesRegex(M.Refusal, "reparse"):
                self.run_tool()
        finally:
            self.remove_directory_alias(link)
        # Use a fresh receipt; initial inventory refusal deliberately leaves its directory.
        self.destination.mkdir()
        link = self.destination / "alias"
        self.make_directory_alias(link, outside)
        try:
            with self.assertRaisesRegex(M.Refusal, "reparse"):
                self.run_tool(dataclasses.replace(self.config, receipt_directory=self.receipt_root / "second"))
        finally:
            self.remove_directory_alias(link)
        self.assertEqual(b"outside", (outside / "untouched.bin").read_bytes())

    def test_only_entire_verified_scope_retired_and_real_alias_resumes_safely(self):
        config = dataclasses.replace(self.config, retire_source=True, junction=True)
        result = self.run_tool(config)
        self.assertEqual("complete", result["status"])
        self.assertTrue(M.reparse(self.source.lstat()))
        self.assertEqual(self.destination, self.source.resolve())
        self.assertEqual(b"first payload", (self.source / "first.bin").read_bytes())
        self.assertTrue(all(item["state"] == "retired" for item in result["files"].values()))
        self.assertEqual(result["logical_source_bytes"], result["logical_retired_bytes"])
        for intent in result["deletion_intents"]:
            self.assertTrue(M.inside(Path(intent["absolute_resolved_path"]), self.source_root))
            self.assertNotEqual(str(self.source_root), intent["absolute_resolved_path"])
        self.assertEqual("complete", self.run_tool(config, resume=True)["status"])
        self.assertEqual(b"second payload with tail", (self.destination / "nested/second.bin").read_bytes())

    def test_retirement_interruption_after_unlink_recovers_without_touching_destination(self):
        config = dataclasses.replace(self.config, retire_source=True, junction=True)
        original = M.Relocator.unlink_source_file
        def unlink_then_fail(operation, path):
            original(operation, path)
            raise OSError("simulated interruption after unlink before receipt commit")
        with patch.object(M.Relocator, "unlink_source_file", unlink_then_fail):
            with self.assertRaisesRegex(OSError, "simulated"):
                self.run_tool(config)
        self.assertFalse((self.source / "first.bin").exists())
        self.assertTrue((self.source / "nested/second.bin").exists())
        self.assertEqual("retire_pending", self.stored()["files"]["first.bin"]["state"])
        self.assertEqual("complete", self.run_tool(config, resume=True)["status"])
        self.assertEqual(b"first payload", (self.destination / "first.bin").read_bytes())

    def test_single_file_existing_verified_retirement_no_alias_and_resume(self):
        source = self.source / "first.bin"
        target = self.destination_root / "one.gguf"
        target.write_bytes(source.read_bytes())
        config = dataclasses.replace(self.config, source=source, destination=target, kind="file", retire_source=True)
        result = self.run_tool(config)
        self.assertEqual("complete", result["status"])
        self.assertFalse(os.path.lexists(source))
        self.assertEqual(b"first payload", target.read_bytes())
        self.assertEqual("complete", self.run_tool(config, resume=True)["status"])
        for extra in ({"junction": True}, {"include_files": ("first.bin",)}):
            with self.assertRaises(M.Refusal):
                M.Relocator(dataclasses.replace(config, **extra))

    def test_alias_failure_after_verified_retirement_can_resume(self):
        config = dataclasses.replace(self.config, retire_source=True, junction=True)
        with patch.object(M.Relocator, "create_alias", side_effect=OSError("injected alias failure")):
            with self.assertRaisesRegex(OSError, "injected alias failure"):
                self.run_tool(config)
        self.assertFalse(os.path.lexists(self.source))
        self.assertEqual(b"first payload", (self.destination / "first.bin").read_bytes())
        self.assertTrue(all(item["state"] == "retired" for item in self.stored()["files"].values()))
        self.assertEqual("complete", self.run_tool(config, resume=True)["status"])
        self.assertEqual(self.destination, self.source.resolve())

    def test_empty_directory_retirement_keeps_root_and_recreates_alias(self):
        empty_source = self.source_root / "empty scope"
        empty_source.mkdir()
        (empty_source / "empty child").mkdir()
        config = dataclasses.replace(self.config, source=empty_source, retire_source=True, junction=True)
        result = self.run_tool(config)
        try:
            self.assertEqual("complete", result["status"])
            self.assertEqual({}, result["files"])
            self.assertTrue(self.source_root.is_dir())
            self.assertEqual(self.destination, empty_source.resolve())
            self.assertTrue((self.destination / "empty child").is_dir())
        finally:
            self.remove_directory_alias(empty_source)

    def test_low_free_space_refuses_before_part_copy_or_source_retirement(self):
        real = shutil.disk_usage
        def low(path):
            actual = real(path)
            return type(actual)(actual.total, actual.total - 1, 1)
        with patch.object(M.shutil, "disk_usage", low):
            with self.assertRaisesRegex(M.Refusal, "Insufficient destination free space"):
                self.run_tool(dataclasses.replace(self.config, retire_source=True))
        self.assertTrue((self.source / "first.bin").exists())
        self.assertFalse((self.destination / "first.bin").exists())
        self.assertEqual([], self.stored()["deletion_intents"])

    def test_new_source_entry_after_copy_blocks_all_retirement(self):
        original = M.Relocator.retirement_preflight
        def add_then_preflight(operation):
            self.add("unexpected.bin", b"new data must remain")
            return original(operation)
        with patch.object(M.Relocator, "retirement_preflight", add_then_preflight):
            with self.assertRaisesRegex(M.Refusal, "gained files"):
                self.run_tool(dataclasses.replace(self.config, retire_source=True))
        self.assertTrue((self.source / "first.bin").exists())
        self.assertTrue((self.source / "unexpected.bin").exists())

    def test_resume_rejects_tampered_receipt_traversal_before_any_file_write(self):
        self.run_tool()
        receipt = self.stored()
        receipt["files"]["../../outside.bin"] = receipt["files"].pop("first.bin")
        (self.receipt / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(M.Refusal, "Invalid relative path"):
            self.run_tool(resume=True)
        self.assertFalse((self.root / "outside.bin").exists())

    @unittest.skipUnless(os.name == "nt", "Windows receipt sharing/access retry policy")
    def test_own_receipt_transient_sharing_retry_is_atomic_and_then_succeeds(self):
        temporary = self.root / "receipt.tmp"
        target = self.root / "receipt.json"
        temporary.write_bytes(b"new receipt")
        target.write_bytes(b"old receipt")
        original = M.os.replace
        calls = []
        def flaky(source, destination):
            calls.append((source, destination))
            if len(calls) <= 2:
                self.assertEqual(b"old receipt", target.read_bytes())
                error = PermissionError("injected reader without FILE_SHARE_DELETE")
                error.winerror = 32 if len(calls) == 1 else 5
                raise error
            return original(source, destination)
        with patch.object(M.os, "replace", flaky), patch.object(M.time, "sleep") as sleep:
            M.replace_own_receipt(temporary, target)
        self.assertEqual(3, len(calls))
        self.assertEqual(2, sleep.call_count)
        self.assertEqual(b"new receipt", target.read_bytes())
        self.assertFalse(temporary.exists())

    @unittest.skipUnless(os.name == "nt", "Windows receipt sharing/access retry policy")
    def test_permanent_receipt_sharing_failure_stops_before_first_source_unlink(self):
        config = dataclasses.replace(self.config, retire_source=True)
        operation = M.Relocator(config)
        original = M.os.replace
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        def refusal_on_deletion_intent(source, destination):
            if operation.receipt.get("deletion_intents"):
                error = PermissionError("permanent injected receipt reader")
                error.winerror = 33
                raise error
            return original(source, destination)
        with patch.object(M.os, "replace", refusal_on_deletion_intent), \
                patch.object(M.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(M.time, "sleep", sleep):
            with self.assertRaisesRegex(PermissionError, "permanent injected"):
                operation.execute()
        # The failure handler must not restart another five-second retry budget.
        self.assertLessEqual(clock[0], 5.001)
        self.assertGreaterEqual(clock[0], 5.0)
        self.assertTrue((self.source / "first.bin").is_file())
        self.assertTrue((self.source / "nested/second.bin").is_file())
        self.assertEqual(b"first payload", (self.destination / "first.bin").read_bytes())
        self.assertEqual([], self.stored()["deletion_intents"])

    def test_receipt_retry_does_not_retry_unrelated_io_or_data_publication(self):
        error = OSError("ordinary unrelated disk failure")
        error.winerror = 112  # ERROR_DISK_FULL is never a sharing retry.
        with patch.object(M.os, "replace", side_effect=error) as replace, patch.object(M.time, "sleep") as sleep:
            with self.assertRaisesRegex(OSError, "unrelated"):
                M.replace_own_receipt(self.root / "receipt.tmp", self.root / "receipt.json")
        self.assertEqual(1, replace.call_count)
        sleep.assert_not_called()
        # The existing publication race test separately proves no-overwrite data rename.


if __name__ == "__main__":
    unittest.main()
