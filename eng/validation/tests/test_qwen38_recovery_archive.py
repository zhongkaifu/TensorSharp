import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "qwen38-recovery-archive.py"
SPEC = importlib.util.spec_from_file_location("recovery_archive", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RecoveryArchiveTests(unittest.TestCase):
    def test_relative_owned_link_chain_captures_every_hop_and_restores_in_fixture_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve();bundle = root / "bundle";owned = root / "owned"
            bundle.mkdir();owned.mkdir();(owned / "libengine.so.1").write_bytes(b"chain bytes")
            (owned / "current.so").symlink_to("libengine.so.1")
            (bundle / "libengine.so").symlink_to("../owned/current.so")
            plan = dict(paths=[str(bundle)], include_symlink_targets_under=[str(owned)])
            result = MODULE.archive_plan(plan, root / "backup")
            rows = json.loads((root / "backup/files.json").read_text())["members"]
            names = {r["source_path"] for r in rows}
            self.assertTrue({str(owned / "current.so"), str(owned / "libengine.so.1")} <= names)
            self.assertEqual((bundle / "libengine.so").read_bytes(), b"chain bytes")
            # The backup tool never restores. This test removes/recreates ONLY its
            # generated temporary fixture, proving the captured relative chain.
            (bundle / "libengine.so").unlink();(owned / "current.so").unlink();(owned / "libengine.so.1").unlink()
            with tarfile.open(result["archive_path"]) as archive:
                for row in rows:
                    target = Path(row["source_path"])
                    self.assertTrue(target.is_relative_to(root))
                    if row["kind"] == "file":
                        target.write_bytes(archive.extractfile(row["archive_path"]).read())
                    elif row["kind"] == "symlink":
                        target.symlink_to(row["target"])
            self.assertEqual((bundle / "libengine.so").read_bytes(), b"chain bytes")
            self.assertEqual(result["tool_version"], "qwen38-recovery-archive/2")

    def test_absolute_then_relative_owned_chain_is_captured(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve();bundle = root / "bundle";owned = root / "owned"
            bundle.mkdir();owned.mkdir();(owned / "actual").write_bytes(b"actual")
            (owned / "middle").symlink_to("actual");(bundle / "entry").symlink_to(owned / "middle")
            rows, _ = MODULE.inventory(dict(paths=[str(bundle)], include_symlink_targets_under=[str(owned)]))
            records = {r["source_path"]: r for r in rows}
            self.assertEqual(records[str(owned / "middle")]["target"], "actual")
            self.assertEqual(records[str(bundle / "entry")]["resolved_target"], str(owned / "actual"))

    def test_external_link_chain_is_recorded_hop_by_hop_but_never_copied(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve();bundle = root / "bundle";system = root / "system"
            bundle.mkdir();system.mkdir();(system / "python3.12").write_bytes(b"external interpreter")
            (system / "python3").symlink_to("python3.12");(bundle / "python").symlink_to(system / "python3")
            rows, dependencies = MODULE.inventory(dict(paths=[str(bundle)], external_dependency_prefixes=[str(system)]))
            self.assertFalse(any(r["source_path"].startswith(str(system)) for r in rows))
            by_path = {r["path"]: r for r in dependencies}
            self.assertEqual(by_path[str(system / "python3")]["kind"], "symlink")
            self.assertEqual(by_path[str(system / "python3")]["target"], "python3.12")
            self.assertEqual(by_path[str(system / "python3.12")]["sha256"], MODULE.sha(system / "python3.12"))
            self.assertTrue(all(r["copied"] is False for r in dependencies))

    def test_external_directory_alias_hop_is_a_restore_prerequisite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve();bundle = root / "bundle";system = root / "system"
            bundle.mkdir();system.mkdir();(system / "lib").mkdir();(system / "lib/file").write_bytes(b"external")
            (system / "alias").symlink_to("lib", target_is_directory=True)
            (bundle / "entry").symlink_to(system / "alias/file")
            _, dependencies = MODULE.inventory(dict(paths=[str(bundle)], external_dependency_prefixes=[str(system)]))
            self.assertIn(str(system / "alias"), {r["path"] for r in dependencies})

    def test_uncovered_or_forbidden_intermediate_cannot_hide_behind_allowed_final_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve();bundle = root / "bundle";owned = root / "owned";other = root / "other"
            bundle.mkdir();owned.mkdir();other.mkdir();(owned / "file").write_bytes(b"allowed final bytes")
            (other / "middle").symlink_to(owned / "file");(bundle / "entry").symlink_to(other / "middle")
            plan = dict(paths=[str(bundle)], include_symlink_targets_under=[str(owned)])
            with self.assertRaisesRegex(ValueError, "Uncovered runtime"):
                MODULE.inventory(plan)
            plan["include_symlink_targets_under"].append(str(other));plan["forbidden_prefixes"] = [str(other)]
            with self.assertRaisesRegex(ValueError, "Forbidden"):
                MODULE.inventory(plan)

    def test_cycle_broken_link_and_owned_directory_need_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve();bundle = root / "bundle";owned = root / "owned"
            bundle.mkdir();owned.mkdir();(bundle / "a").symlink_to("b");(bundle / "b").symlink_to("a")
            with self.assertRaisesRegex(ValueError, "cycle"):
                MODULE.inventory(dict(paths=[str(bundle)]))
            (bundle / "b").unlink();(bundle / "b").symlink_to("absent")
            with self.assertRaises(FileNotFoundError):
                MODULE.inventory(dict(paths=[str(bundle)]))
            (bundle / "a").unlink();(bundle / "b").unlink();(bundle / "file").write_bytes(b"not a directory")
            (bundle / "bad-slash").symlink_to("file/")
            with self.assertRaises(OSError):
                MODULE.inventory(dict(paths=[str(bundle)]))
            (bundle / "bad-slash").unlink();(bundle / "directory").symlink_to(owned)
            with self.assertRaisesRegex(ValueError, "explicit plan root"):
                MODULE.inventory(dict(paths=[str(bundle)], include_symlink_targets_under=[str(owned)]))

    def test_target_dotdot_resolves_directory_alias_before_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve();bundle = root / "bundle";owned = root / "owned"
            bundle.mkdir();owned.mkdir();(owned / "nested/sub").mkdir(parents=True)
            (owned / "nested/selected").write_bytes(b"correct POSIX target")
            (owned / "selected").write_bytes(b"wrong lexical normpath target")
            (owned / "alias").symlink_to("nested/sub", target_is_directory=True)
            (bundle / "entry").symlink_to("../owned/alias/../selected")
            rows, _ = MODULE.inventory(dict(paths=[str(bundle), str(owned)]))
            entry = next(r for r in rows if r["source_path"] == str(bundle / "entry"))
            self.assertEqual(entry["resolved_target"], str(owned / "nested/selected"))

    def test_runtime_link_target_outside_bundle_is_included_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);bundle = root / "bundle";build = root / "build"
            bundle.mkdir();build.mkdir();target = build / "libengine.so.1"
            target.write_bytes(b"actual runtime bytes")
            (bundle / "libengine.so").symlink_to(target)
            plan = dict(paths=[str(bundle)], include_symlink_targets_under=[str(build)])
            result = MODULE.archive_plan(plan, root / "backup")
            self.assertEqual(result["status"], "VERIFIED_SOURCE_UNCHANGED")
            self.assertEqual(result["verified_regular_file_members"], 1)
            self.assertEqual(target.read_bytes(), b"actual runtime bytes")
            self.assertTrue((bundle / "libengine.so").is_symlink())

    def test_uncovered_or_forbidden_model_link_cannot_claim_restorable_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);bundle = root / "bundle";bundle.mkdir()
            model = root / "weights";model.write_bytes(b"not a runtime")
            (bundle / "link").symlink_to(model)
            with self.assertRaisesRegex(ValueError, "Uncovered runtime"):
                MODULE.inventory(dict(paths=[str(bundle)]))
            with self.assertRaisesRegex(ValueError, "Forbidden"):
                MODULE.inventory(dict(paths=[str(bundle)], forbidden_prefixes=[str(model)]))

    def test_hardlinked_files_are_verified_and_source_root_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);bundle = root / "bundle";bundle.mkdir()
            (bundle / "a").write_bytes(b"same bytes")
            import os
            os.link(bundle / "a", bundle / "b")
            result = MODULE.archive_plan(dict(paths=[str(bundle)]), root / "backup")
            self.assertEqual(result["verified_regular_file_members"], 2)
            with self.assertRaisesRegex(ValueError, "destination"):
                MODULE.archive_plan(dict(paths=[str(bundle)]), bundle / "bad")

    def test_corrupt_archive_content_cannot_pass_full_member_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);source = root / "runtime";source.write_bytes(b"expected")
            rows, _ = MODULE.inventory(dict(paths=[str(source)]))
            archive = root / "corrupt.tar"
            with tarfile.open(archive, "w") as output:
                info = output.gettarinfo(str(source), arcname=rows[0]["archive_path"])
                output.addfile(info, io.BytesIO(b"tampered"))
            with self.assertRaisesRegex(ValueError, "content differs"):
                MODULE.verify_archive(archive, rows)


if __name__ == "__main__":
    unittest.main()
