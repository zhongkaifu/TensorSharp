#!/usr/bin/env python3
"""Resume Apple notarization and finish a DMG without an Installer certificate.

Keep OUTPUT in ignored artifacts/ or docs/validation/. The persistent private
stage and Apple receipts intentionally survive failures and timeouts. Rerun the
same command to resume; an ambiguous submit without a receipt never resubmits.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import re
import subprocess
import sys
import uuid


_spec = importlib.util.spec_from_file_location(
    "tensoragent_signing", Path(__file__).with_name("verify-tensoragent-macos-signing.py"))
signing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(signing)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_digest(app):
    digest = hashlib.sha256()
    for path in sorted(app.rglob("*")):
        digest.update(str(path.relative_to(app)).encode() + b"\0")
        if path.is_symlink():
            digest.update(b"link\0" + str(path.readlink()).encode())
        elif path.is_file():
            digest.update(sha256(path).encode())
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def accepted(result, returncode):
    if returncode != 0 or result.get("status") != "Accepted":
        raise ValueError(f"Notarization was not accepted: {result.get('status', 'unknown')}")


class Finalizer:
    def __init__(self, options, run=subprocess.run, audit=signing.audit):
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?(?:\+[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?", options.version):
            raise ValueError("Expected a three-part release version without a leading v")
        self.options, self.run, self.audit = options, run, audit
        self.app = Path(options.app).resolve()
        self.output = Path(options.output).resolve()
        self.stem = f"tensoragent-desktop-{options.version}-osx-arm64"
        self.stage = self.output / ("." + self.stem + "-stage")
        self.stage.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.stage.chmod(0o700)
        self.evidence = self.stage / "evidence"
        self.evidence.mkdir(exist_ok=True)
        self.checkpoint = self.stage / "checkpoint.json"
        self.state = read_json(self.checkpoint) if self.checkpoint.exists() else {}
        self.notary = ["--keychain-profile", options.notary_profile]
        self.disk = self.stage / "disk"
        self.staged_app = self.disk / "TensorAgent.app"
        self.dmg = self.stage / (self.stem + ".dmg")
        self.final = self.output / self.dmg.name

    def save(self):
        write_json(self.checkpoint, self.state)

    def command(self, label, args, check=True):
        print(f"{label}: {' '.join(args[:3])}", flush=True)
        result = self.run(args, capture_output=True, text=True, check=False)
        (self.evidence / (label + ".stdout.txt")).write_text(result.stdout or "", encoding="utf-8")
        (self.evidence / (label + ".stderr.txt")).write_text(result.stderr or "", encoding="utf-8")
        write_json(self.evidence / (label + ".result.json"), {"returncode": result.returncode})
        if check and result.returncode:
            raise RuntimeError(f"{label} failed ({result.returncode}); see {self.evidence}")
        return result

    def audit_app(self, app, label):
        print(f"{label}: auditing Developer ID signatures and entitlements", flush=True)

        def runner(args, **kwargs):
            result = self.command(label + "-" + str(self.state.get("audit_calls", 0)), args)
            self.state["audit_calls"] = self.state.get("audit_calls", 0) + 1
            return result

        return self.audit(app, run=runner)

    def validate_payload(self):
        if not self.app.is_dir() or self.app.name != "TensorAgent.app":
            raise ValueError("--app must name an existing TensorAgent.app")
        version = self.options.version
        info = plistlib.loads((self.app / "Contents/Info.plist").read_bytes())
        expected = ".".join(str(int(part)) for part in re.split(r"[-+]", version)[0].split("."))
        if info.get("CFBundleShortVersionString") != expected:
            raise ValueError(f"App version does not match {expected}")

        def os_version(value):
            return tuple((list(map(int, value.split("."))) + [0, 0, 0])[:3])

        if os_version(info["LSMinimumSystemVersion"]) > (14, 0, 0):
            raise ValueError("App exceeds the documented macOS 14 floor")
        self.main_relative = Path("Contents/MacOS") / info["CFBundleExecutable"]
        executable = self.app / self.main_relative
        library = self.app / "Contents/MonoBundle/libGgmlOps.dylib"
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("Missing executable main application binary")
        for label, binary in (("main-arm64", executable), ("native-arm64", library)):
            self.command(label, ["lipo", str(binary), "-verify_arch", "arm64"])
        build = self.command("native-build", ["xcrun", "vtool", "-show-build", str(library)]).stdout
        minimum = re.search(r"\bminos\s+([0-9.]+)", build)
        if not minimum or os_version(minimum[1]) > (14, 0, 0):
            raise ValueError("Native library exceeds the documented macOS 14 floor")
        dependencies = self.command("native-dependencies", ["otool", "-L", str(library)]).stdout.splitlines()[2:]
        for line in dependencies:
            dependency = line.strip().split(" ")[0]
            if not dependency.startswith(("/System/Library/", "/usr/lib/")):
                raise ValueError(f"Unbundled native dependency: {dependency}")
        for relative in ("Contents/Resources/webui/index.html", "Contents/Resources/skills"):
            if not (self.app / relative).exists():
                raise ValueError(f"Missing app payload: {relative}")

    def wait(self, submission_id, label):
        uuid.UUID(submission_id)
        receipt = self.evidence / (label + "-wait.json")
        run_receipt = self.evidence / (label + "-wait.result.json")
        cached_wait = (receipt.exists() and read_json(receipt).get("status") == "Accepted"
                       and run_receipt.exists() and read_json(run_receipt).get("returncode") == 0)
        if not cached_wait:
            response = self.command(label + "-wait", ["xcrun", "notarytool", "wait", submission_id,
                                    *self.notary, "--timeout", self.options.timeout, "--output-format", "json"],
                                    check=False)
            try:
                result = json.loads(response.stdout)
            except ValueError as error:
                raise ValueError(f"No valid {label} wait receipt; see {self.evidence}") from error
            write_json(receipt, result)
            returncode = response.returncode
        else:
            result, returncode = read_json(receipt), 0
        if str(uuid.UUID(result["id"])) != str(uuid.UUID(submission_id)):
            raise ValueError("Apple wait response belongs to a different submission")
        if result.get("status") not in ("Accepted", "Invalid", "Rejected"):
            accepted(result, returncode)
        log = self.evidence / (label + "-log.json")

        def matching_accepted_log(document):
            try:
                same_job = uuid.UUID(document.get("jobId", "")) == uuid.UUID(submission_id)
            except (ValueError, AttributeError, TypeError):
                return False
            return same_job and document.get("status") == "Accepted" and document.get("statusCode") == 0

        try:
            document = read_json(log) if log.exists() else {}
        except ValueError:
            document = {}
        if not (cached_wait and matching_accepted_log(document)):
            self.command(label + "-log", ["xcrun", "notarytool", "log", submission_id, *self.notary, str(log)])
            document = read_json(log)
        if result.get("status") == "Accepted" and not matching_accepted_log(document):
            raise ValueError("Apple log does not match the Accepted submission")
        issues = document.get("issues") or []
        write_json(self.evidence / (label + "-warnings.json"),
                   [issue for issue in issues if issue.get("severity", "").lower() == "warning"])
        accepted(result, returncode)
        print(f"{label}: Apple Accepted ({submission_id})", flush=True)

    def staple(self, path, label):
        self.command(label + "-staple", ["xcrun", "stapler", "staple", str(path)])
        self.command(label + "-ticket", ["xcrun", "stapler", "validate", str(path)])

    def execute(self):
        with (self.stage / "lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Another finalizer is using this stage; do not run concurrent workers") from error
            self.state = read_json(self.checkpoint) if self.checkpoint.exists() else {}
            return self._execute()

    def _execute(self):
        self.validate_payload()
        team = self.audit_app(self.app, "input-app")
        configuration = {"app": str(self.app), "version": self.options.version,
                         "identity": self.options.identity, "notary_profile": self.options.notary_profile,
                         "app_submission_id": str(uuid.UUID(self.options.app_submission_id)),
                         "source_digest": bundle_digest(self.app),
                         "source_main_sha256": sha256(self.app / self.main_relative), "team": team}
        if self.state.get("configuration") not in (None, configuration):
            raise ValueError("Checkpoint belongs to a different signed app or configuration; use another output directory")
        self.state["configuration"] = configuration
        self.save()
        if self.state.get("validated_sha256"):
            candidate = self.dmg if self.dmg.exists() else self.final
            if not candidate.exists() or sha256(candidate) != self.state["validated_sha256"]:
                raise ValueError("Validated DMG was removed or changed")
            return self.promote(candidate)
        self.wait(configuration["app_submission_id"], "app")
        if bundle_digest(self.app) != configuration["source_digest"]:
            raise ValueError("Signed source app changed while awaiting notarization")
        if not self.state.get("app_copied"):
            self.disk.mkdir(exist_ok=True)
            self.command("copy-app", ["ditto", str(self.app), str(self.staged_app)])
            if bundle_digest(self.staged_app) != configuration["source_digest"]:
                raise ValueError("Copied app differs from the audited signed source")
            applications = self.disk / "Applications"
            if not applications.exists() and not applications.is_symlink():
                applications.symlink_to("/Applications")
            self.state["app_copied"] = True
            self.save()
        if sha256(self.staged_app / self.main_relative) != configuration["source_main_sha256"]:
            raise ValueError("Staged main binary differs from the audited signed source")
        self.staple(self.staged_app, "app")
        self.audit_app(self.staged_app, "staged-app")
        self.command("app-gatekeeper", ["spctl", "--assess", "--type", "execute", "--verbose=4", str(self.staged_app)])
        (self.disk / "INSTALL.txt").write_text(
            "TensorAgent Desktop for Apple Silicon (macOS 14 or later)\n\n"
            "Drag TensorAgent.app to Applications, eject this disk, then open TensorAgent.\n"
            "Quit TensorAgent before updating. Models and chats are stored outside the app.\n"
            "Developer ID signed and Apple-notarized; app and disk tickets are stapled.\n"
            "Models are separate downloads. Python/Node.js are optional tools for skills.\n"
            "https://tensorsharp.ai/tensoragent.html\n", encoding="utf-8")
        if not self.state.get("dmg_signed"):
            self.state["embedded_app_digest"] = bundle_digest(self.staged_app)
            self.save()
            self.command("create-dmg", ["hdiutil", "create", "-volname", "TensorAgent", "-srcfolder",
                         str(self.disk), "-format", "UDZO", "-ov", str(self.dmg)])
            self.command("sign-dmg", ["codesign", "--sign", self.options.identity, "--timestamp", str(self.dmg)])
            self.state["dmg_signed"] = sha256(self.dmg)
            self.save()
        self.verify_dmg_checkpoint()
        submission = self.evidence / "dmg-submit.json"
        if not self.state.get("dmg_submission_id"):
            if self.state.get("dmg_submit_attempted"):
                if not submission.exists():
                    raise ValueError("DMG submit may have reached Apple without a receipt; recover its submission ID before resuming")
                self.state["dmg_submission_id"] = str(uuid.UUID(read_json(submission)["id"]))
                self.save()
            else:
                self.state["dmg_submit_attempted"] = True
                self.save()
                response = self.command("dmg-submit", ["xcrun", "notarytool", "submit", str(self.dmg),
                                        *self.notary, "--output-format", "json"], check=False)
                receipt = json.loads(response.stdout)
                write_json(submission, receipt)
                self.state["dmg_submission_id"] = str(uuid.UUID(receipt["id"]))
                self.save()
                if response.returncode:
                    raise RuntimeError("DMG submission returned an error; receipt retained for resumption")
        self.wait(self.state["dmg_submission_id"], "dmg")
        self.state["dmg_accepted"] = True
        self.save()
        # The wait can last hours. Verify the exact submitted identity again.
        self.verify_dmg_checkpoint()
        self.staple(self.dmg, "dmg")
        self.state["dmg_stapled"] = sha256(self.dmg)
        self.save()
        self.command("dmg-gatekeeper", ["spctl", "--assess", "--type", "open", "--context",
                     "context:primary-signature", "--verbose=4", str(self.dmg)])
        self.command("verify-disk-image", ["hdiutil", "verify", str(self.dmg)])
        self.verify_mounted_app()
        if sha256(self.dmg) != self.state["dmg_stapled"]:
            raise ValueError("Stapled DMG changed during final validation")
        self.state["validated_sha256"] = self.state["dmg_stapled"]
        self.save()
        return self.promote(self.dmg)

    def verify_dmg_checkpoint(self):
        self.command("verify-dmg-signature", ["codesign", "--verify", "--strict", "--test-requirement",
                     signing.DEVELOPER_ID_REQUIREMENT, str(self.dmg)])
        metadata = self.command("dmg-signature-metadata", ["codesign", "--display", "--verbose=4", str(self.dmg)]).stderr
        signing.validate_metadata(metadata, self.dmg, team=self.state["configuration"]["team"])
        match = re.search(r"^CDHash=([0-9a-fA-F]+)$", metadata, re.MULTILINE)
        if not match:
            raise ValueError("DMG signature has no CodeDirectory identity")
        cdhash = match[1].lower()
        if self.state.get("dmg_cdhash") not in (None, cdhash):
            raise ValueError("DMG signed payload differs from the submitted artifact")
        self.state["dmg_cdhash"] = cdhash
        current = sha256(self.dmg)
        expected = self.state.get("dmg_stapled", self.state["dmg_signed"])
        if current != expected:
            if not self.state.get("dmg_accepted") or self.state.get("dmg_stapled"):
                raise ValueError("Submitted staged DMG changed; refusing to reuse its receipt")
            # Recover the narrow crash window between staple success and save:
            # unchanged verified CodeDirectory plus an independently valid ticket.
            self.command("recover-dmg-ticket", ["xcrun", "stapler", "validate", str(self.dmg)])
            self.state["dmg_stapled"] = current
        self.save()

    def verify_mounted_app(self):
        mountpoint = self.stage / "mount"
        mountpoint.mkdir(exist_ok=True)
        mounts = [mountpoint]
        response = self.command("mount-dmg", ["hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint",
                                str(mountpoint), "-plist", str(self.dmg)],
                                check=False)
        try:
            entities = plistlib.loads(response.stdout.encode())["system-entities"]
            reported = [Path(entity["mount-point"]).resolve() for entity in entities if "mount-point" in entity]
            mounts += [path for path in reported if path != mountpoint]
            if response.returncode or reported != [mountpoint]:
                raise ValueError("DMG did not mount as exactly one readable volume")
            app = mountpoint / "TensorAgent.app"
            self.audit_app(app, "mounted-app")
            if bundle_digest(app) != self.state["embedded_app_digest"]:
                raise ValueError("Mounted app differs from the expected signed payload")
            self.command("mounted-app-ticket", ["xcrun", "stapler", "validate", str(app)])
            self.command("mounted-app-gatekeeper", ["spctl", "--assess", "--type", "execute", "--verbose=4", str(app)])
        finally:
            failures = []
            for mount in mounts:
                result = self.command("detach-dmg", ["hdiutil", "detach", str(mount)], check=False)
                if result.returncode:
                    failures.append(str(mount))
            if failures and not response.returncode:
                raise RuntimeError(f"Could not detach mounted DMG: {', '.join(failures)}")

    def promote(self, candidate):
        checksum = self.stage / ("SHA256SUMS-" + self.stem + ".txt")
        checksum.write_text(self.state["validated_sha256"] + "  " + self.dmg.name + "\n", encoding="utf-8")
        if candidate != self.final:
            candidate.replace(self.final)
        checksum.replace(self.output / checksum.name)
        self.state["complete"] = True
        self.save()
        print(f"Validated and promoted: {self.final}", flush=True)
        return self.final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("app", "version", "identity", "notary-profile", "app-submission-id", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--timeout", default="48h")
    options = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("DMG finalization requires macOS")
    try:
        Finalizer(options).execute()
    except (ValueError, RuntimeError, OSError, KeyError, plistlib.InvalidFileException) as error:
        parser.exit(1, f"DMG finalization failed: {error}\n")


if __name__ == "__main__":
    main()
