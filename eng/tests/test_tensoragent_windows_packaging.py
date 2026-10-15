"""Contract checks for desktop MSI authoring; no Windows installation is claimed."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


ENG = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tensoragent_wix", ENG / "generate-tensoragent-wix.py")
wix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wix)
NS = {"w": wix.WIX_NAMESPACE}


class WindowsDesktopPackagingTests(unittest.TestCase):
    def setUp(self):
        artifact_root = ENG.parent / "artifacts"
        artifact_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="windows-packaging-fixture-", dir=artifact_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.publish = self.root / "publish"
        for name in wix.REQUIRED_FILES + ("skills/example/SKILL.md", "webui/assets/a&b.js", ".hidden-payload"):
            target = self.publish / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture:" + name.encode("utf-8"))
        self.write_runtime({"includedFrameworks": [{"name": "Microsoft.NETCore.App", "version": "10.0.0"}]})

    def write_runtime(self, options):
        (self.publish / "TensorAgent.Maui.runtimeconfig.json").write_text(json.dumps({"runtimeOptions": options}), encoding="utf-8")

    def test_version_triplets_and_calendar_mapping(self):
        for release, numeric in (("2.8.6", "2.8.6"), ("2026.10.03", "26.10.3"),
                                 ("2026.10.03-preview.1+build.2", "26.10.3"), ("2255.255.65535", "255.255.65535")):
            with self.subTest(release=release):
                self.assertEqual(wix.msi_version(release), numeric)

    def test_reject_invalid_or_unrepresentable_versions(self):
        for version in ("v2.8.6", "2.8", "2.8.6.1", "2.8.6-", "2.8.6..preview", "2/8/6",
                        "2.256.1", "2.0.65536", "256.0.0", "2256.1.1", "2.8.6\n", "2.8.6+$oops"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                wix.msi_version(version)

    def test_payload_manifest_preserves_hidden_nested_and_escaped_files(self):
        files = wix.payload_manifest(self.publish, "win-x64-cpu")
        by_path = {item["path"]: item for item in files}
        for name in (".hidden-payload", "webui/assets/a&b.js", "skills/example/SKILL.md"):
            self.assertIn(name, by_path)
            self.assertEqual(by_path[name]["sha256"], hashlib.sha256((self.publish / name).read_bytes()).hexdigest())

    def test_missing_runtime_native_webui_or_crt_fails(self):
        for name in ("coreclr.dll", "GgmlOps.dll", "Microsoft.UI.Xaml.dll", "webui/index.html", "vcruntime140.dll"):
            target = self.publish / name
            content = target.read_bytes()
            target.unlink()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "missing/empty"):
                wix.payload_manifest(self.publish, "win-x64-cpu")
            target.write_bytes(content)

    def test_missing_skills_fails(self):
        (self.publish / "skills/example/SKILL.md").unlink()
        with self.assertRaisesRegex(ValueError, "no bundled skills"):
            wix.payload_manifest(self.publish, "win-x64-cpu")

    def test_framework_dependent_runtime_fails(self):
        self.write_runtime({"framework": {"name": "Microsoft.NETCore.App", "version": "10.0.0"}})
        with self.assertRaisesRegex(ValueError, "framework-dependent"):
            wix.payload_manifest(self.publish, "win-x64-cpu")

    def test_cuda_must_bundle_runtime_and_ptx(self):
        with self.assertRaisesRegex(ValueError, "cudart64_12.dll"):
            wix.payload_manifest(self.publish, "win-x64-cuda")
        for name in wix.CUDA_FILES:
            target = self.publish / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"cuda fixture")
        self.assertTrue(wix.payload_manifest(self.publish, "win-x64-cuda"))
        (self.publish / "cuda_kernels/tensorsharp_kernels.ptx").unlink()
        with self.assertRaisesRegex(ValueError, "tensorsharp_kernels.ptx"):
            wix.payload_manifest(self.publish, "win-x64-cuda")

    def test_symbolic_links_are_rejected(self):
        try:
            (self.publish / "linked.dll").symlink_to(self.publish / "GgmlOps.dll")
        except OSError as error:
            if sys.platform == "win32" and getattr(error, "winerror", None) == 1314:
                self.skipTest("Windows symbolic-link creation requires Developer Mode or the symbolic-link privilege.")
            raise
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            wix.payload_manifest(self.publish, "win-x64-cpu")

    def test_windows_invalid_names_are_rejected(self):
        # Do not physically create these names. On Windows a colon denotes an
        # NTFS stream, while Win32 removes trailing dots/spaces from filenames.
        names = [f"bad{character}name.txt" for character in '<>:"\\|?*']
        names += [f"bad{chr(value)}name.txt" for value in range(32)]
        names += ["trailing.", "trailing ", "trailing. ", "webui/trailing./index.html"]
        for name in names:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Invalid Windows"):
                wix.validate_windows_relative_path(name)

    def test_windows_reserved_devices_are_rejected_in_every_component(self):
        devices = ["CON", "PRN", "AUX", "NUL"]
        devices += [f"{prefix}{digit}" for prefix in ("COM", "LPT") for digit in "123456789¹²³"]
        for device in devices:
            for name in (device, device.lower() + ".txt", device + ".tar.gz", device + " .txt", "skills/" + device + "/SKILL.md"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Invalid Windows"):
                    wix.validate_windows_relative_path(name)

    def test_windows_payload_paths_must_be_canonical_and_relative(self):
        for name in ("", ".", "..", "./index.html", "../index.html", "webui/../index.html",
                     "webui/./index.html", "/index.html", "//server/share/index.html", "C:/index.html",
                     "C:index.html", "\\index.html", "\\\\server\\share\\index.html", "webui//index.html", "webui/"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Invalid Windows"):
                wix.validate_windows_relative_path(name)

    def test_windows_valid_payload_names_are_accepted_without_normalization(self):
        for name in ("TensorAgent.Maui.exe", ".hidden-payload", "webui/assets/a&b.js", "skills/example.v2/SKILL.md",
                     "COM10.txt", "LPT10/index.html", "webui/conversation.txt", "日本語/model.json", "version 1.0/data.json"):
            with self.subTest(name=name):
                self.assertIsNone(wix.validate_windows_relative_path(name))

    def test_payload_scan_applies_lexical_validation_to_files_and_directories(self):
        # Synthetic traversal entries preserve the invalid spelling on Windows
        # without creating ADS/device files or relying on POSIX-only filenames.
        for relative in ("webui/bad:name.txt", "skills/trailing. ", "skills/AUX/example.md"):
            with self.subTest(relative=relative), patch.object(Path, "rglob", return_value=[self.publish / relative]):
                with self.assertRaisesRegex(ValueError, "Invalid Windows"):
                    wix.payload_manifest(self.publish, "win-x64-cpu")

    def test_both_variants_share_upgrade_family_and_replace_same_version(self):
        files = wix.payload_manifest(self.publish, "win-x64-cpu")
        for variant in wix.VARIANTS:
            tree = wix.generate_wix(self.publish, "2026.10.03-preview.1", variant, files)
            # Serialize and parse to check XML escaping, including ampersands.
            root = ET.fromstring(ET.tostring(tree.getroot()))
            package = root.find("w:Package", NS)
            self.assertEqual(package.get("UpgradeCode"), wix.UPGRADE_CODE)
            self.assertEqual(package.get("Scope"), "perUser")
            self.assertEqual(package.get("Version"), "26.10.3")
            upgrade = package.find("w:MajorUpgrade", NS)
            self.assertEqual(upgrade.get("AllowSameVersionUpgrades"), "yes")
            self.assertEqual(upgrade.get("Schedule"), "afterInstallInitialize")
            installed = root.findall(".//w:File", NS)
            self.assertEqual(len(installed), len(files))
            self.assertEqual({item.get("Source") for item in installed}, {str(self.publish / item["path"]) for item in files})
            self.assertEqual(package.find("w:Property[@Id='ARPHELPLINK']", NS).get("Value"), "https://tensorsharp.ai/tensoragent.html")
            shortcut = root.find(".//w:Shortcut", NS)
            self.assertEqual(shortcut.get("Target"), "[INSTALLFOLDER]TensorAgent.Maui.exe")
            self.assertEqual(shortcut.get("WorkingDirectory"), "INSTALLFOLDER")
            self.assertEqual(len(root.findall(".//w:Component", NS)), len(root.findall(".//w:ComponentRef", NS)))
            self.assertTrue(all(item.find("w:RegistryValue", NS).get("Root") == "HKCU" for item in root.findall(".//w:Component", NS)))

    def test_component_ids_and_guids_are_stable_across_release_and_variant(self):
        files = wix.payload_manifest(self.publish, "win-x64-cpu")
        first = wix.generate_wix(self.publish, "2.8.6", "win-x64-cpu", files)
        second = wix.generate_wix(self.publish, "2.8.7", "win-x64-cuda", files)
        components = lambda tree: {(item.get("Id"), item.get("Guid")) for item in tree.findall(".//w:Component", NS)}
        self.assertEqual(components(first), components(second))

    def test_cli_creates_manifest_and_refuses_generated_files_in_payload(self):
        command = [sys.executable, str(ENG / "generate-tensoragent-wix.py"), "--publish-directory", str(self.publish),
                   "--version", "2026.10.03", "--variant", "win-x64-cpu", "--manifest", str(self.root / "payload.json")]
        result = subprocess.run(command + ["--output", str(self.root / "TensorAgent.wxs")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.root / "payload.json").read_text())["msi_version"], "26.10.3")
        result = subprocess.run(command + ["--output", str(self.publish / "TensorAgent.wxs")], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the publish directory", result.stderr)


if __name__ == "__main__":
    unittest.main()
