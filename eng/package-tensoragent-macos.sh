#!/usr/bin/env bash
# Package a Developer ID signed Mac Catalyst app; notarize every distribution.
# Usage: bash eng/package-tensoragent-macos.sh APP VERSION [OUTPUT_DIRECTORY] [--ad-hoc]
# --ad-hoc is only for local testing, never for public releases.
set -euo pipefail

APP="${1:?Supply the path to TensorAgent.app}"
VERSION="${2:?Supply the release version without a leading v}"
OUTPUT="${3:-artifacts}"
MODE="${4:-}"
[[ $# -le 4 && ( -z "$MODE" || "$MODE" == --ad-hoc ) ]] || {
    echo "Usage: $0 APP VERSION [OUTPUT_DIRECTORY] [--ad-hoc]" >&2; exit 1;
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ "$(uname -s)" == Darwin ]] || { echo "macOS packaging requires macOS." >&2; exit 1; }
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z]+([.-][0-9A-Za-z]+)*)?(\+[0-9A-Za-z]+([.-][0-9A-Za-z]+)*)?$ ]] || {
    echo "Expected a three-part release version (for example 2026.10.03 or 2.8.6)." >&2; exit 1;
}
[[ -d "$APP" && "$(basename "$APP")" == TensorAgent.app ]] || { echo "TensorAgent.app not found: $APP" >&2; exit 1; }
for file in Contents/Info.plist Contents/MonoBundle/libGgmlOps.dylib Contents/Resources/webui/index.html; do
    [[ -s "$APP/$file" ]] || { echo "Missing app payload: $file" >&2; exit 1; }
done
[[ -d "$APP/Contents/Resources/skills" ]] || { echo "Missing bundled skills." >&2; exit 1; }
EXECUTABLE="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$APP/Contents/Info.plist")"
[[ -x "$APP/Contents/MacOS/$EXECUTABLE" ]] || { echo "Missing app executable." >&2; exit 1; }
codesign --verify --deep --strict "$APP"
if [[ "$MODE" != --ad-hoc ]]; then
    [[ "${TENSORAGENT_APPLICATION_IDENTITY:-}" == "Developer ID Application: "* ]] || {
        echo "Set TENSORAGENT_APPLICATION_IDENTITY to a Developer ID Application identity." >&2; exit 1;
    }
    [[ "${TENSORAGENT_INSTALLER_IDENTITY:-}" == "Developer ID Installer: "* ]] || {
        echo "Set TENSORAGENT_INSTALLER_IDENTITY to a Developer ID Installer identity." >&2; exit 1;
    }
    [[ -n "${TENSORAGENT_NOTARY_PROFILE:-}" ]] || {
        echo "Set TENSORAGENT_NOTARY_PROFILE to credentials stored by xcrun notarytool store-credentials." >&2; exit 1;
    }
    python3 "$SCRIPT_DIR/verify-tensoragent-macos-signing.py" "$APP"
else
    echo "LOCAL TEST PACKAGE: ad-hoc signing does not satisfy Gatekeeper; do not publish." >&2
fi
# Xcode 26.6 treats every argument after -verify_arch as an architecture.
lipo "$APP/Contents/MacOS/$EXECUTABLE" -verify_arch arm64
lipo "$APP/Contents/MonoBundle/libGgmlOps.dylib" -verify_arch arm64
python3 - "$APP" "$VERSION" <<'PY'
import plistlib
from pathlib import Path
import re
import subprocess
import sys

app = Path(sys.argv[1])
info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
expected = ".".join(str(int(part)) for part in re.split(r"[-+]", sys.argv[2])[0].split("."))
if info.get("CFBundleShortVersionString") != expected:
    raise SystemExit(f"App version {info.get('CFBundleShortVersionString')} does not match release {expected}")
def os_version(value):
    parts = list(map(int, value.split(".")))
    return tuple((parts + [0, 0, 0])[:3])

if os_version(info["LSMinimumSystemVersion"]) > (14, 0, 0):
    raise SystemExit("The published app requires newer than the documented macOS 14 floor")
library = app / "Contents/MonoBundle/libGgmlOps.dylib"
build = subprocess.check_output(["xcrun", "vtool", "-show-build", str(library)], text=True)
minimum = re.search(r"\bminos\s+([0-9.]+)", build)
if not minimum or os_version(minimum[1]) > (14, 0, 0):
    raise SystemExit("GGML native library must be built with MACOSX_DEPLOYMENT_TARGET=14.0")
dependencies = subprocess.check_output(["otool", "-L", str(library)], text=True).splitlines()[2:]
for line in dependencies:
    name = line.strip().split(" ")[0]
    if not name.startswith(("/System/Library/", "/usr/lib/")):
        raise SystemExit(f"Unbundled GGML native dependency: {name}")
PY

mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
STEM="tensoragent-desktop-$VERSION-osx-arm64"
WORK="$(mktemp -d "$OUTPUT/.tensoragent-macos.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/disk" "$WORK/root/Applications" "$WORK/packages"
PACKAGES="$WORK/packages"
ditto "$APP" "$WORK/disk/TensorAgent.app"
ln -s /Applications "$WORK/disk/Applications"
if [[ "$MODE" != --ad-hoc ]]; then
    mkdir -p "$OUTPUT/notarization"
    NOTARY_ARGS=(--keychain-profile "$TENSORAGENT_NOTARY_PROFILE")
    KEYCHAIN_ARGS=()
    if [[ -n "${TENSORAGENT_SIGNING_KEYCHAIN:-}" ]]; then
        NOTARY_ARGS+=(--keychain "$TENSORAGENT_SIGNING_KEYCHAIN")
        KEYCHAIN_ARGS=(--keychain "$TENSORAGENT_SIGNING_KEYCHAIN")
    fi

    notarize() {
        local file="$1" label="$2" result="$OUTPUT/notarization/$STEM-$2-submit.json"
        local log="$OUTPUT/notarization/$STEM-$2-log.json" status=0 submission_id
        xcrun notarytool submit "$file" "${NOTARY_ARGS[@]}" --wait \
            --timeout "${TENSORAGENT_NOTARY_TIMEOUT:-30m}" --output-format json > "$result" || status=$?
        submission_id="$(python3 - "$result" <<'PY'
import json
from pathlib import Path
import sys
import uuid

try:
    print(uuid.UUID(json.loads(Path(sys.argv[1]).read_text())["id"]))
except (ValueError, KeyError):
    pass
PY
)"
        if [[ -n "$submission_id" ]]; then
            xcrun notarytool log "$submission_id" "${NOTARY_ARGS[@]}" "$log" \
                || echo "Could not retrieve notarization log for $label ($submission_id)." >&2
        fi
        # notarytool may exit successfully with an Invalid submission status.
        # An upload or a timeout must never count as accepted notarization.
        python3 - "$result" "$status" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
try:
    result = json.loads(path.read_text())
except ValueError:
    raise SystemExit(f"Notary service returned no valid result; see {path}")
if sys.argv[2] != "0" or result.get("status") != "Accepted" or not result.get("id"):
    raise SystemExit(f"Notarization was not accepted ({result.get('status', 'unknown')}); see {path}")
print(f"Accepted notarization: {result['id']}")
PY
    }

    staple() {
        xcrun stapler staple "$1"
        xcrun stapler validate "$1"
    }

    # ZIP has no place for a ticket; staple the app before making final archives.
    ditto -c -k --sequesterRsrc --keepParent "$WORK/disk/TensorAgent.app" "$WORK/notarize-app.zip"
    notarize "$WORK/notarize-app.zip" app
    staple "$WORK/disk/TensorAgent.app"
    spctl --assess --type execute --verbose=4 "$WORK/disk/TensorAgent.app"
fi
cat > "$WORK/disk/INSTALL.txt" <<'EOF'
TensorAgent Desktop for Apple Silicon (macOS 14 or later)

DMG: drag TensorAgent.app to Applications, eject the disk, then open TensorAgent.
ZIP: extract, move TensorAgent.app to Applications, then open TensorAgent.
PKG: open the installer and follow the prompts to install into /Applications.
Quit TensorAgent before updating. Models and chats are stored outside the app.

Open menu > Models, choose a model that fits your RAM, Download, then Use.
Model files are separate multi-GB downloads; .NET and Metal/CPU are bundled.
Python/Node.js are optional system tools for code and browser skills.

Download, installation, first chat and troubleshooting guide:
https://tensorsharp.ai/tensoragent.html
https://github.com/zhongkaifu/TensorSharp/releases
EOF
if [[ "$MODE" == --ad-hoc ]]; then
    cat >> "$WORK/disk/INSTALL.txt" <<'EOF'

LOCAL TEST BUILD: ad-hoc signed, not Developer ID signed or notarized.
macOS will block this build under normal Gatekeeper policy. For a build you trust,
attempt to open it, then use System Settings > Privacy & Security > Open Anyway.
Do not distribute these local test packages as public releases.
EOF
else
    echo 'Developer ID signed and Apple-notarized; tickets are stapled for offline installation.' >> "$WORK/disk/INSTALL.txt"
fi

# ditto preserves executable permissions, symlinks and the bundle signature.
ditto -c -k --sequesterRsrc --keepParent "$WORK/disk/TensorAgent.app" "$PACKAGES/$STEM.zip"
hdiutil create -volname TensorAgent -srcfolder "$WORK/disk" -format UDZO -ov "$PACKAGES/$STEM.dmg"
ditto "$WORK/disk/TensorAgent.app" "$WORK/root/Applications/TensorAgent.app"
# Explicitly disable bundle relocation: Installer must update /Applications,
# rather than a developer's bin/ tree or a copy mounted on the DMG.
pkgbuild --analyze --root "$WORK/root" "$WORK/components.plist"
python3 - "$WORK/components.plist" <<'PY'
import plistlib
from pathlib import Path
import sys

path = Path(sys.argv[1])
components = plistlib.loads(path.read_bytes())
if not any(component.get("RootRelativeBundlePath") == "Applications/TensorAgent.app" for component in components):
    raise SystemExit("pkgbuild did not find the TensorAgent application bundle")
for component in components:
    # Newer pkgbuild versions omit this default-valued key from --analyze.
    component["BundleIsRelocatable"] = False
path.write_bytes(plistlib.dumps(components))
PY
PKG_VERSION="${VERSION%%[-+]*}"
pkgbuild --root "$WORK/root" --component-plist "$WORK/components.plist" \
    --identifier ai.tensorsharp.tensoragent.desktop --version "$PKG_VERSION" \
    --install-location / "$WORK/component.pkg"
cat > "$WORK/requirements.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>os</key><array><string>14.0</string></array>
<key>arch</key><array><string>arm64</string></array>
</dict></plist>
EOF
PRODUCT_SIGN_ARGS=()
if [[ "$MODE" != --ad-hoc ]]; then
    # macOS still ships Bash 3.2, where empty arrays are unbound under set -u.
    PRODUCT_SIGN_ARGS=(--sign "$TENSORAGENT_INSTALLER_IDENTITY" --timestamp ${KEYCHAIN_ARGS[@]+"${KEYCHAIN_ARGS[@]}"})
fi
productbuild --product "$WORK/requirements.plist" --package "$WORK/component.pkg" \
    ${PRODUCT_SIGN_ARGS[@]+"${PRODUCT_SIGN_ARGS[@]}"} "$PACKAGES/$STEM.pkg"

if [[ "$MODE" != --ad-hoc ]]; then
    codesign --sign "$TENSORAGENT_APPLICATION_IDENTITY" --timestamp \
        ${KEYCHAIN_ARGS[@]+"${KEYCHAIN_ARGS[@]}"} "$PACKAGES/$STEM.dmg"
    codesign --verify --strict "$PACKAGES/$STEM.dmg"
    pkgutil --check-signature "$PACKAGES/$STEM.pkg"
    notarize "$PACKAGES/$STEM.dmg" dmg
    staple "$PACKAGES/$STEM.dmg"
    spctl --assess --type open --context context:primary-signature --verbose=4 "$PACKAGES/$STEM.dmg"
    notarize "$PACKAGES/$STEM.pkg" pkg
    staple "$PACKAGES/$STEM.pkg"
    spctl --assess --type install --verbose=4 "$PACKAGES/$STEM.pkg"
fi

# Validate the actual archives, including the signature after ZIP extraction.
mkdir -p "$WORK/extracted"
ditto -x -k "$PACKAGES/$STEM.zip" "$WORK/extracted"
codesign --verify --deep --strict "$WORK/extracted/TensorAgent.app"
if [[ "$MODE" != --ad-hoc ]]; then
    python3 "$SCRIPT_DIR/verify-tensoragent-macos-signing.py" "$WORK/extracted/TensorAgent.app"
    xcrun stapler validate "$WORK/extracted/TensorAgent.app"
    spctl --assess --type execute --verbose=4 "$WORK/extracted/TensorAgent.app"
fi
hdiutil verify "$PACKAGES/$STEM.dmg"
pkgutil --expand "$PACKAGES/$STEM.pkg" "$WORK/pkg-expanded"
test -s "$WORK/pkg-expanded/Distribution"
(
    cd "$PACKAGES"
    shasum -a 256 "$STEM.dmg" "$STEM.pkg" "$STEM.zip" > "SHA256SUMS-$STEM.txt"
    shasum -a 256 -c "SHA256SUMS-$STEM.txt"
)
# Promote only complete, validated artifacts. Diagnostics remain under ignored artifacts/.
for file in "$STEM.dmg" "$STEM.pkg" "$STEM.zip" "SHA256SUMS-$STEM.txt"; do
    mv "$PACKAGES/$file" "$OUTPUT/$file"
done
echo "Packaged TensorAgent Desktop $VERSION for osx-arm64 in $OUTPUT (mode: ${MODE:-Developer ID + notarization})"
