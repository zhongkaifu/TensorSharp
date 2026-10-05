"""Release versions and platform packaging commands must satisfy their contracts."""
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import shlex
import subprocess
import sys
import tempfile
import unittest

ENG = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resolver = load("release_version", ENG / "resolve-release-version.py")
wix = load("wix_version", ENG / "generate-tensoragent-wix.py")
signing = load("macos_signing", ENG / "verify-tensoragent-macos-signing.py")


def signing_metadata(*, authority="Developer ID Application: TensorSharp (ABC1234567)",
                     team="ABC1234567", timestamp=True, runtime=True):
    lines = ["Identifier=ai.tensorsharp.tensoragent", f"Authority={authority}",
             f"TeamIdentifier={team}", "flags=0x10000(runtime)" if runtime else "flags=0x0(none)"]
    if timestamp:
        lines.append("Timestamp=Oct 3, 2026 at 12:00:00 PM")
    return "\n".join(lines)


class MacSigningVerificationTests(unittest.TestCase):
    def test_developer_id_metadata_requires_secure_timestamp_and_hardened_runtime(self):
        self.assertEqual(signing.validate_metadata(signing_metadata(), Path("TensorAgent.app"),
                                                   require_runtime=True), "ABC1234567")
        for changes in ({"timestamp": False}, {"runtime": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                signing.validate_metadata(signing_metadata(**changes), Path("TensorAgent.app"),
                                          require_runtime=True)

    def test_adhoc_and_app_store_distribution_are_not_developer_id(self):
        for metadata in ("Signature=adhoc\nTeamIdentifier=not set\n",
                         signing_metadata(authority="Apple Distribution: TensorSharp (ABC1234567)")):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                signing.validate_metadata(metadata, Path("TensorAgent.app"))

    def test_local_signing_time_does_not_replace_secure_timestamp(self):
        metadata = signing_metadata(timestamp=False) + "\nSigned Time=Oct 3, 2026 at 12:00:00 PM"
        with self.assertRaises(ValueError):
            signing.validate_metadata(metadata, Path("TensorAgent.app"))

    def test_native_code_must_have_same_team_and_timestamp(self):
        self.assertEqual(signing.validate_metadata(signing_metadata(runtime=False), Path("libGgmlOps.dylib"),
                                                   team="ABC1234567"), "ABC1234567")
        for changes in ({"team": "XYZ1234567"}, {"timestamp": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                signing.validate_metadata(signing_metadata(**changes), Path("libGgmlOps.dylib"),
                                          team="ABC1234567")

    def test_jit_entitlement_is_required_and_debug_and_sandbox_are_rejected(self):
        valid = {"com.apple.security.cs.allow-jit": True}
        signing.validate_entitlements(valid)
        for entitlements in ({}, {"com.apple.security.cs.allow-jit": False},
                             dict(valid, **{"get-task-allow": True}),
                             dict(valid, **{"com.apple.security.get-task-allow": True}),
                             dict(valid, **{"com.apple.security.app-sandbox": True})):
            with self.subTest(entitlements=entitlements), self.assertRaises(ValueError):
                signing.validate_entitlements(entitlements)

    def create_app(self, directory):
        app = Path(directory) / "TensorAgent.app"
        executable = app / "Contents/MacOS/TensorAgent.Maui"
        library = app / "Contents/MonoBundle/libGgmlOps.dylib"
        framework = app / "Contents/Frameworks/Example.framework/Versions/A/Example"
        for binary in (executable, library, framework):
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(b"\xcf\xfa\xed\xfe" + b"mock Mach-O payload")
        (app / "Contents/Info.plist").write_bytes(plistlib.dumps({
            "CFBundleExecutable": executable.name,
            "CFBundleIdentifier": "ai.tensorsharp.tensoragent",
        }))
        return app, executable, library, framework

    def codesign_runner(self, metadata=None, entitlements=None):
        calls = []
        metadata = metadata or {}
        if entitlements is None:
            entitlements = {"com.apple.security.cs.allow-jit": True}

        def run(command, **kwargs):
            calls.append(command)
            self.assertEqual(kwargs, {"check": True, "capture_output": True, "text": True})
            self.assertEqual(command[0], "codesign")
            if "--entitlements" in command:
                self.assertIn("--xml", command, "codesign defaults to a human-readable entitlement dump")
                return subprocess.CompletedProcess(command, 0, plistlib.dumps(entitlements).decode(), "")
            if "--display" in command:
                details = metadata.get(Path(command[-1]), signing_metadata())
                return subprocess.CompletedProcess(command, 0, "", details)
            self.assertIn("--verify", command)
            return subprocess.CompletedProcess(command, 0, "", "")

        return run, calls

    def test_audit_verifies_developer_id_requirement_for_every_macho(self):
        with tempfile.TemporaryDirectory() as directory:
            app, executable, library, framework = self.create_app(directory)
            run, calls = self.codesign_runner()
            self.assertEqual(signing.audit(app, run=run), "ABC1234567")
            verified = {Path(command[-1]) for command in calls if "--verify" in command}
            self.assertEqual(verified, {app, executable, library, framework})
            for command in calls:
                if "--verify" in command:
                    self.assertIn("--test-requirement", command)
                    requirement = command[command.index("--test-requirement") + 1]
                    self.assertTrue(requirement.startswith("="), "codesign inline requirements need the '=' prefix")
                    self.assertIn("anchor apple generic", requirement)
                    self.assertIn("1.2.840.113635.100.6.1.13", requirement)

    def test_audit_rejects_native_library_signed_by_another_team(self):
        with tempfile.TemporaryDirectory() as directory:
            app, _, library, _ = self.create_app(directory)
            run, _ = self.codesign_runner(metadata={library: signing_metadata(team="XYZ1234567")})
            with self.assertRaises(ValueError):
                signing.audit(app, run=run)

    def test_audit_requires_runtime_on_the_actual_main_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            app, executable, _, _ = self.create_app(directory)
            run, _ = self.codesign_runner(metadata={executable: signing_metadata(runtime=False)})
            with self.assertRaises(ValueError):
                signing.audit(app, run=run)

    def test_audit_reads_and_checks_app_entitlements(self):
        with tempfile.TemporaryDirectory() as directory:
            app, _, _, _ = self.create_app(directory)
            run, _ = self.codesign_runner(entitlements={"com.apple.security.cs.allow-jit": True,
                                                       "com.apple.security.get-task-allow": True})
            with self.assertRaises(ValueError):
                signing.audit(app, run=run)


class ReleaseVersionTests(unittest.TestCase):
    def test_calendar_tag_preserves_asset_name_and_normalizes_app_metadata(self):
        self.assertEqual(resolver.resolve("2026.10.03"), ("2026.10.03", "2026.10.3"))
        self.assertEqual(wix.msi_version("2026.10.03"), "26.10.3")

    def test_prerelease_and_build_label_stay_in_asset_name(self):
        version = "2.8.6-rc.1+build.2"
        self.assertEqual(resolver.resolve(version), (version, "2.8.6"))
        self.assertEqual(wix.msi_version(version), "2.8.6")

    def test_unrepresentable_installer_versions_fail_before_building(self):
        for version in ("256.1.0", "2026.256.0", "2.8.65536", "2256.1.1"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                resolver.resolve(version)

    def test_manual_input_cannot_inject_actions_outputs_or_paths(self):
        for version in ("2.8.6\ntag=v0.0.0", "../../2.8.6", "2.8.6$(id)", "v2.8.6", "", "2.8"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                resolver.resolve(version)

    def test_tag_dispatch_emits_consistent_actions_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            environment = dict(os.environ, RELEASE_VERSION="", GITHUB_REF_TYPE="tag",
                               GITHUB_REF_NAME="v2026.10.03", GITHUB_OUTPUT=str(output))
            subprocess.run([sys.executable, str(ENG / "resolve-release-version.py")],
                           env=environment, check=True, capture_output=True, text=True)
            self.assertEqual(output.read_text(), "version=2026.10.03\ntag=v2026.10.03\napp_version=2026.10.3\n")


@unittest.skipUnless(sys.platform == "darwin", "Mac packager uses macOS bundle tools")
class MacDesktopPackagingTests(unittest.TestCase):
    def packaging_fixture(self, root):
        app = root / "TensorAgent.app"
        executable = app / "Contents/MacOS/TensorAgent.Maui"
        library = app / "Contents/MonoBundle/libGgmlOps.dylib"
        for binary in (executable, library):
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(b"\xcf\xfa\xed\xfe" + b"mock Mach-O payload")
        executable.chmod(0o755)
        resources = app / "Contents/Resources"
        (resources / "webui").mkdir(parents=True)
        (resources / "webui/index.html").write_text("fixture", encoding="utf-8")
        (resources / "skills").mkdir()
        (app / "Contents/Info.plist").write_bytes(plistlib.dumps({
            "CFBundleExecutable": executable.name,
            "CFBundleIdentifier": "ai.tensorsharp.tensoragent",
            "CFBundleShortVersionString": "2026.10.3",
            "LSMinimumSystemVersion": "14.0",
        }))
        commands = root / "commands"
        commands.mkdir()
        log = root / "commands.jsonl"
        shim = root / "mac_tools.py"
        # These mocks test control flow and archive contents, never actual Apple
        # signing, notarization, Gatekeeper, or installer validity.
        shim.write_text('''import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import zipfile

name = Path(sys.argv[1]).name
args = sys.argv[2:]
with open(os.environ["PACKAGING_COMMAND_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps({"command": name, "args": args}) + "\\n")
if name == "codesign":
    if "--entitlements" in args:
        sys.stdout.buffer.write(plistlib.dumps({"com.apple.security.cs.allow-jit": True}))
    elif "--display" in args:
        print("Identifier=ai.tensorsharp.tensoragent\\nAuthority=Developer ID Application: TensorSharp (ABC1234567)\\nTeamIdentifier=ABC1234567\\nTimestamp=Oct 3, 2026 at 12:00:00 PM\\nflags=0x10000(runtime)", file=sys.stderr)
elif name == "xcrun":
    if args[0] == "vtool":
        print("minos 14.0")
    elif args[:2] == ["notarytool", "submit"]:
        status = "Invalid" if os.environ.get("PACKAGING_REJECT_SUFFIX") and args[2].endswith(os.environ["PACKAGING_REJECT_SUFFIX"]) else "Accepted"
        print(json.dumps({"id": "11111111-2222-4333-8444-555555555555", "status": status}))
    elif args[:2] == ["notarytool", "log"]:
        Path(args[-1]).write_text(json.dumps({"issues": []}))
    elif args[:2] == ["stapler", "staple"]:
        if os.environ.get("PACKAGING_FAIL_STAPLE_SUFFIX") and args[-1].endswith(os.environ["PACKAGING_FAIL_STAPLE_SUFFIX"]):
            sys.exit(65)
        if Path(args[-1]).is_dir():
            (Path(args[-1]) / "Contents/mock-stapled-ticket").write_text("ticket")
    elif args[0] not in ("notarytool", "stapler"):
        sys.exit("Unhandled xcrun command: " + repr(args))
elif name == "otool":
    print(args[-1] + ":\\n\\t@rpath/libGgmlOps.dylib (compatibility version 0.0.0)\\n\\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0)")
elif name == "ditto":
    if "-c" in args:
        source, destination = map(Path, args[-2:])
        with zipfile.ZipFile(destination, "w") as archive:
            for path in source.rglob("*"):
                archive.write(path, path.relative_to(source.parent))
    elif "-x" in args:
        with zipfile.ZipFile(args[-2]) as archive:
            archive.extractall(args[-1])
    else:
        shutil.copytree(args[-2], args[-1], symlinks=True, dirs_exist_ok=True)
elif name == "hdiutil":
    if args[0] == "create":
        Path(args[-1]).write_bytes(b"mock disk image")
elif name == "pkgbuild":
    if "--analyze" in args:
        Path(args[-1]).write_bytes(plistlib.dumps([{"RootRelativeBundlePath": "Applications/TensorAgent.app"}]))
    else:
        Path(args[-1]).write_bytes(b"mock component package")
elif name == "productbuild":
    Path(args[-1]).write_bytes(b"mock product package")
elif name == "pkgutil":
    if args[0] == "--expand":
        destination = Path(args[-1])
        destination.mkdir()
        (destination / "Distribution").write_text("mock distribution")
    else:
        print("Status: signed by a certificate trusted by macOS\\nCertificate Chain:\\n 1. Developer ID Installer: TensorSharp (ABC1234567)")
elif name == "shasum":
    sys.exit(subprocess.run(["/usr/bin/shasum", *args]).returncode)
elif name not in ("lipo", "spctl"):
    sys.exit("Unhandled mock tool: " + name)
''', encoding="utf-8")
        for name in ("codesign", "lipo", "xcrun", "otool", "ditto", "hdiutil", "pkgbuild",
                     "productbuild", "pkgutil", "spctl", "shasum"):
            command = commands / name
            command.write_text(
                f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(shim))} \"$0\" \"$@\"\n",
                encoding="utf-8")
            command.chmod(0o755)
        environment = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"],
                           PACKAGING_COMMAND_LOG=str(log),
                           TENSORAGENT_APPLICATION_IDENTITY="Developer ID Application: TensorSharp (ABC1234567)",
                           TENSORAGENT_INSTALLER_IDENTITY="Developer ID Installer: TensorSharp (ABC1234567)",
                           TENSORAGENT_NOTARY_PROFILE="mock-notary-profile")
        return app, environment, log

    def package(self, app, root, environment, *options):
        return subprocess.run(["bash", str(ENG / "package-tensoragent-macos.sh"), str(app), "2026.10.03",
                               str(root / "packages"), *options], env=environment,
                              capture_output=True, text=True)

    def test_default_release_requires_credentials_before_building_archives(self):
        for missing in ("TENSORAGENT_APPLICATION_IDENTITY", "TENSORAGENT_INSTALLER_IDENTITY",
                        "TENSORAGENT_NOTARY_PROFILE"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                app, environment, log = self.packaging_fixture(root)
                environment.pop(missing)
                result = self.package(app, root, environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(missing, result.stderr)
                self.assertFalse(list((root / "packages").glob("*.zip")))

    def test_invalid_notary_response_even_with_exit_zero_does_not_publish_archives(self):
        for rejected_suffix in (".zip", ".dmg", ".pkg"):
            with self.subTest(rejected_suffix=rejected_suffix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                app, environment, log = self.packaging_fixture(root)
                environment["PACKAGING_REJECT_SUFFIX"] = rejected_suffix
                result = self.package(app, root, environment)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("Invalid", result.stdout + result.stderr)
                output = root / "packages"
                for pattern in ("*.zip", "*.dmg", "*.pkg", "SHA256SUMS-*"):
                    self.assertFalse(list(output.glob(pattern)))
                submissions = list((output / "notarization").glob("*-submit.json"))
                self.assertEqual(len(submissions), (".zip", ".dmg", ".pkg").index(rejected_suffix) + 1)
                self.assertEqual(sum(json.loads(path.read_text())["status"] == "Invalid"
                                     for path in submissions), 1)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertFalse(any(call["command"] == "shasum" for call in calls))
                self.assertFalse(any(call["command"] == "xcrun"
                                     and call["args"][:2] == ["stapler", "staple"]
                                     and Path(call["args"][-1]).suffix == (".app" if rejected_suffix == ".zip"
                                                                            else rejected_suffix)
                                     for call in calls))

    def test_accepted_app_is_stapled_before_archives_and_containers_before_checksums(self):
        import zipfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, environment, log = self.packaging_fixture(root)
            result = self.package(app, root, environment)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = root / "packages"
            for suffix in (".zip", ".dmg", ".pkg"):
                self.assertEqual(len(list(output.glob("*" + suffix))), 1, result.stdout + result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            submits = [(index, call["args"][2]) for index, call in enumerate(calls)
                       if call["command"] == "xcrun" and call["args"][:2] == ["notarytool", "submit"]]
            self.assertEqual([Path(path).suffix for _, path in submits], [".zip", ".dmg", ".pkg"])
            for index, _ in submits:
                args = calls[index]["args"]
                self.assertIn("--wait", args)
                self.assertEqual(args[args.index("--keychain-profile") + 1], "mock-notary-profile")
            staples = [(index, call["args"][-1]) for index, call in enumerate(calls)
                       if call["command"] == "xcrun" and call["args"][:2] == ["stapler", "staple"]]
            self.assertEqual([Path(path).suffix for _, path in staples], [".app", ".dmg", ".pkg"])
            checksum_index = next(index for index, call in enumerate(calls) if call["command"] == "shasum")
            self.assertGreater(checksum_index, max(index for index, _ in staples))
            for (submit_index, _), (staple_index, _) in zip(submits, staples):
                self.assertLess(submit_index, staple_index)
            archive_index = next(index for index, call in enumerate(calls)
                                 if call["command"] == "ditto" and "-c" in call["args"]
                                 and Path(call["args"][-1]).name.startswith("tensoragent-desktop-"))
            self.assertLess(staples[0][0], archive_index)
            with zipfile.ZipFile(next(output.glob("*.zip"))) as archive:
                self.assertIn("TensorAgent.app/Contents/mock-stapled-ticket", archive.namelist())
            self.assertEqual(len(list(output.glob("SHA256SUMS-*"))), 1)

    def test_failed_container_staple_does_not_promote_archives_or_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, environment, log = self.packaging_fixture(root)
            environment["PACKAGING_FAIL_STAPLE_SUFFIX"] = ".dmg"
            result = self.package(app, root, environment)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            output = root / "packages"
            for pattern in ("*.zip", "*.dmg", "*.pkg", "SHA256SUMS-*"):
                self.assertFalse(list(output.glob(pattern)))
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertFalse(any(call["command"] == "shasum" for call in calls))

    def test_explicit_ad_hoc_mode_does_not_use_notary_or_signing_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, environment, log = self.packaging_fixture(root)
            for credential in ("TENSORAGENT_APPLICATION_IDENTITY", "TENSORAGENT_INSTALLER_IDENTITY",
                               "TENSORAGENT_NOTARY_PROFILE"):
                environment.pop(credential)
            result = self.package(app, root, environment, "--ad-hoc")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = root / "packages"
            for suffix in (".zip", ".dmg", ".pkg"):
                self.assertEqual(len(list(output.glob("*" + suffix))), 1, result.stdout + result.stderr)
            self.assertIn("LOCAL TEST PACKAGE", result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertFalse(any(call["command"] == "xcrun" and call["args"][0] in ("notarytool", "stapler")
                                 for call in calls))
            self.assertFalse(any("--sign" in call["args"] for call in calls))

    def test_checks_both_binaries_with_xcode26_lipo_argument_order(self):
        artifact_root = ENG.parent / "artifacts"
        artifact_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mac packaging fixture ", dir=artifact_root) as directory:
            root = Path(directory)
            app = root / "TensorAgent.app"
            executable = app / "Contents/MacOS/TensorAgent.Maui"
            library = app / "Contents/MonoBundle/libGgmlOps.dylib"
            for file in (executable, library, app / "Contents/Resources/webui/index.html"):
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"fixture")
            executable.chmod(0o755)
            (app / "Contents/Resources/skills").mkdir()
            (app / "Contents/Info.plist").write_bytes(plistlib.dumps({"CFBundleExecutable": executable.name}))

            commands = root / "commands"
            commands.mkdir()
            log = root / "lipo.jsonl"
            # Reproduce Xcode 26.6's grammar: all arguments after -verify_arch
            # are architecture names. Stop after both calls, before packaging;
            # real signatures and archives are validated separately on macOS.
            shim = root / "strict_lipo.py"
            shim.write_text('''import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if "-verify_arch" not in args:
    sys.exit(64)
command = args.index("-verify_arch")
if command == 0 or args[command + 1:] != ["arm64"]:
    sys.exit(64)
if not all(Path(source).is_file() for source in args[:command]):
    sys.exit(65)
with open(os.environ["LIPO_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(args[:command]) + "\\n")
if any(source.endswith("libGgmlOps.dylib") for source in args[:command]):
    sys.exit(73)
''', encoding="utf-8")
            (commands / "codesign").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            (commands / "lipo").write_text(
                f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(shim))} \"$@\"\n",
                encoding="utf-8")
            for command in commands.iterdir():
                command.chmod(0o755)
            environment = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"], LIPO_LOG=str(log))
            result = subprocess.run(["bash", str(ENG / "package-tensoragent-macos.sh"), str(app), "2026.10.03",
                                     str(root / "packages"), "--ad-hoc"], env=environment,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 73, result.stderr)
            self.assertEqual([json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()],
                             [[str(executable)], [str(library)]])


if __name__ == "__main__":
    unittest.main()
