#!/usr/bin/env python3
"""Reject Mac release bundles that cannot be submitted to Apple's notary service."""
from pathlib import Path
import plistlib
import re
import subprocess
import sys


DEVELOPER_ID_REQUIREMENT = (
    # codesign treats a requirement without '=' as a file path.
    "=anchor apple generic and "
    "certificate leaf[field.1.2.840.113635.100.6.1.13] exists"
)
MACHO_MAGIC = {
    bytes.fromhex(value) for value in (
        "feedface", "cefaedfe", "feedfacf", "cffaedfe",
        "cafebabe", "bebafeca", "cafebabf", "bfbafeca",
    )
}


def validate_metadata(metadata, path, team=None, require_runtime=False):
    if "Signature=adhoc" in metadata or not re.search(
            r"^Authority=Developer ID Application: .+$", metadata, re.MULTILINE):
        raise ValueError(f"{path}: a Developer ID Application signature is required")
    match = re.search(r"^TeamIdentifier=([A-Z0-9]{10})$", metadata, re.MULTILINE)
    if not match or (team is not None and match[1] != team):
        raise ValueError(f"{path}: missing or mismatched Developer ID team")
    if not re.search(r"^Timestamp=.+$", metadata, re.MULTILINE):
        raise ValueError(f"{path}: a secure signing timestamp is required")
    flags = re.search(r"\bflags=0x([0-9a-fA-F]+)", metadata)
    if require_runtime and (not flags or not int(flags[1], 16) & 0x10000):
        raise ValueError(f"{path}: publish with UseHardenedRuntime=true")
    return match[1]


def validate_entitlements(entitlements):
    for name in ("get-task-allow", "com.apple.security.get-task-allow",
                 "com.apple.security.app-sandbox"):
        if entitlements.get(name):
            raise ValueError(f"Release app must not grant {name}")
    if entitlements.get("com.apple.security.cs.allow-jit") is not True:
        raise ValueError("The .NET runtime requires com.apple.security.cs.allow-jit")


def audit(app, run=subprocess.run):
    app = Path(app)

    def codesign(*args):
        return run(["codesign", *args], check=True, capture_output=True, text=True)

    codesign("--verify", "--deep", "--strict", "--test-requirement",
             DEVELOPER_ID_REQUIREMENT, str(app))
    metadata = codesign("--display", "--verbose=4", str(app)).stderr
    team = validate_metadata(metadata, app, require_runtime=True)
    if not re.search(r"^Identifier=ai\.tensorsharp\.tensoragent$", metadata, re.MULTILINE):
        raise ValueError("Unexpected TensorAgent bundle identifier")
    # Current codesign defaults to a human-readable representation, not a plist.
    entitlements = codesign("--display", "--entitlements", "-", "--xml", str(app)).stdout
    validate_entitlements(plistlib.loads(entitlements.encode()))
    # Inspect every shipped Mach-O, including NativeReference dylibs and helpers.
    # Verification of the root alone does not require nested code's Developer ID.
    for path in sorted((app / "Contents").rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        with path.open("rb") as stream:
            if stream.read(4) not in MACHO_MAGIC:
                continue
        codesign("--verify", "--strict", "--test-requirement",
                 DEVELOPER_ID_REQUIREMENT, str(path))
        validate_metadata(codesign("--display", "--verbose=4", str(path)).stderr,
                          path, team=team,
                          require_runtime=path.parent == app / "Contents/MacOS")
    return team


if __name__ == "__main__":
    try:
        print(f"Verified Developer ID signatures for team {audit(Path(sys.argv[1]))}")
    except (ValueError, plistlib.InvalidFileException, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise SystemExit(f"Mac release signing validation failed: {detail}") from error
