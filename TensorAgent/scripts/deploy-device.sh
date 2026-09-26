#!/usr/bin/env bash
# Builds, signs, installs, and launches a device build of TensorAgent on a
# connected physical iPhone. Existing app data is preserved: devicectl installs
# the new bundle over the old one and this script never uninstalls the app.
#
# Defaults are deliberately strict. A single connected physical iOS device and
# a single Apple Development identity can be selected automatically; ambiguity
# is an error rather than a reason to deploy to an arbitrary phone.
#
# Env overrides:
#   CONFIGURATION         Debug (default) | Release. Both carry the engine on a
#                         device: Release keeps the dynamically resolved GGML
#                         exports through the ReferenceNativeSymbol list in
#                         TensorAgent.Maui/GgmlExportedSymbols.targets, and this
#                         script refuses to install an executable that lost them
#                         (it looks for _TSGgml_IsMetalAvailable).
#   DEVICE_ID             CoreDevice identifier, hardware UDID, or exact name
#   CODESIGN_KEY          Apple Development identity (auto-selected if unique)
#   CODESIGN_PROVISION    development profile name or UUID for the containing app
#                         (auto-selected if omitted)
#   CODESIGN_SHARE_PROVISION  independent development profile name or UUID for
#                         ai.tensorsharp.tensoragent.share (auto-selected if omitted)
#   TENSORAGENT_SHARE_EXTENSION  true (default) | false. When false, deploy the
#                         intentionally app-only bundle and do not require App Groups.
#   SKIP_LAUNCH=1         install the app without launching it
#   DEVICECTL_TIMEOUT     install timeout in seconds (default: 300)
#   TENSORAGENT_REBUILD_XCFRAMEWORK=0  reuse the existing native xcframework;
#                         the default is 1 so the device build includes current sources
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIGURATION="${CONFIGURATION:-Debug}"
APP="${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/bin/${CONFIGURATION}/net10.0-ios/ios-arm64/TensorAgent.Maui.app"
BUNDLE_ID="ai.tensorsharp.tensoragent"
SHARE_BUNDLE_ID="ai.tensorsharp.tensoragent.share"
APP_GROUP="group.ai.tensorsharp.tensoragent"
ENTITLEMENTS="${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/Platforms/iOS/Entitlements.plist"
DEVICECTL_TIMEOUT="${DEVICECTL_TIMEOUT:-300}"
TENSORAGENT_REBUILD_XCFRAMEWORK="${TENSORAGENT_REBUILD_XCFRAMEWORK:-1}"
TENSORAGENT_SHARE_EXTENSION="${TENSORAGENT_SHARE_EXTENSION:-true}"
REQUESTED_DEVICE="${DEVICE_ID:-}"

case "${CONFIGURATION}" in
    Debug|Release) ;;
    *)
        echo "deploy-device: CONFIGURATION must be Debug or Release" >&2
        exit 1
        ;;
esac

case "${TENSORAGENT_SHARE_EXTENSION}" in
    1|true|TRUE|yes|YES) TENSORAGENT_SHARE_EXTENSION=true ;;
    0|false|FALSE|no|NO) TENSORAGENT_SHARE_EXTENSION=false ;;
    *)
        echo "deploy-device: TENSORAGENT_SHARE_EXTENSION must be true or false" >&2
        exit 1
        ;;
esac

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensoragent-deploy.XXXXXX")"

cleanup() {
    # TEMP_DIR is a resolved directory created by mktemp above. Avoid a recursive
    # broad delete: this script creates only flat temporary files here.
    find "${TEMP_DIR}" -type f -delete 2>/dev/null || true
    rmdir "${TEMP_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

fail() {
    echo "deploy-device: $*" >&2
    exit 1
}

plist_value() { # file key-path
    plutil -extract "$2" raw -o - "$1" 2>/dev/null || true
}

plist_array_contains() { # file PlistBuddy-key-path exact-value
    local file="$1" key="$2" expected="$3" index=0 value
    while value="$(/usr/libexec/PlistBuddy -c "Print :${key}:${index}" "${file}" 2>/dev/null)"; do
        [[ "${value}" == "${expected}" ]] && return 0
        index=$((index + 1))
    done
    return 1
}

require_tool() {
    command -v "$1" >/dev/null 2>&1 || fail "required tool '$1' was not found"
}

for TOOL in xcrun plutil security codesign nm base64 shasum; do
    require_tool "${TOOL}"
done
xcrun --find devicectl >/dev/null 2>&1 || fail "devicectl is unavailable; install/select a current Xcode"

[[ "${DEVICECTL_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || fail "DEVICECTL_TIMEOUT must be a positive integer"

# Apple's supported scripting interface for devicectl is its versioned JSON
# output. Restrict discovery to physical iOS devices with an active CoreDevice
# tunnel; paired-but-offline phones and watches are not deployment candidates.
DEVICES_JSON="${TEMP_DIR}/devices.json"
xcrun devicectl --quiet --timeout 30 --json-output "${DEVICES_JSON}" \
    list devices \
    --filter "hardwareProperties.platform == 'iOS' AND hardwareProperties.reality == 'physical' AND connectionProperties.tunnelState == 'connected'"

DEVICE_COUNT="$(plist_value "${DEVICES_JSON}" result.devices)"
[[ "${DEVICE_COUNT}" =~ ^[0-9]+$ ]] || fail "could not read devicectl's device list"

print_devices() {
    local index=0 name identifier udid
    while [[ "${index}" -lt "${DEVICE_COUNT}" ]]; do
        name="$(plist_value "${DEVICES_JSON}" "result.devices.${index}.deviceProperties.name")"
        identifier="$(plist_value "${DEVICES_JSON}" "result.devices.${index}.identifier")"
        udid="$(plist_value "${DEVICES_JSON}" "result.devices.${index}.hardwareProperties.udid")"
        printf '  %s (identifier %s, UDID %s)\n' "${name:-unknown}" "${identifier:-unknown}" "${udid:-unknown}" >&2
        index=$((index + 1))
    done
}

SELECTED_INDEX=""
if [[ -n "${REQUESTED_DEVICE}" ]]; then
    INDEX=0
    while [[ "${INDEX}" -lt "${DEVICE_COUNT}" ]]; do
        CANDIDATE_IDENTIFIER="$(plist_value "${DEVICES_JSON}" "result.devices.${INDEX}.identifier")"
        CANDIDATE_UDID="$(plist_value "${DEVICES_JSON}" "result.devices.${INDEX}.hardwareProperties.udid")"
        CANDIDATE_NAME="$(plist_value "${DEVICES_JSON}" "result.devices.${INDEX}.deviceProperties.name")"
        if [[ "${REQUESTED_DEVICE}" == "${CANDIDATE_IDENTIFIER}" ||
              "${REQUESTED_DEVICE}" == "${CANDIDATE_UDID}" ||
              "${REQUESTED_DEVICE}" == "${CANDIDATE_NAME}" ]]; then
            [[ -z "${SELECTED_INDEX}" ]] || fail "DEVICE_ID '${REQUESTED_DEVICE}' matches more than one connected device"
            SELECTED_INDEX="${INDEX}"
        fi
        INDEX=$((INDEX + 1))
    done
    if [[ -z "${SELECTED_INDEX}" ]]; then
        echo "Connected physical iOS devices:" >&2
        print_devices
        fail "DEVICE_ID '${REQUESTED_DEVICE}' is not a connected physical iOS device"
    fi
else
    if [[ "${DEVICE_COUNT}" -ne 1 ]]; then
        echo "Connected physical iOS devices:" >&2
        print_devices
        fail "found ${DEVICE_COUNT}; connect exactly one or set DEVICE_ID"
    fi
    SELECTED_INDEX=0
fi

DEVICE_NAME="$(plist_value "${DEVICES_JSON}" "result.devices.${SELECTED_INDEX}.deviceProperties.name")"
DEVICE_UDID="$(plist_value "${DEVICES_JSON}" "result.devices.${SELECTED_INDEX}.hardwareProperties.udid")"
DEVICE_IDENTIFIER="$(plist_value "${DEVICES_JSON}" "result.devices.${SELECTED_INDEX}.identifier")"
DEVELOPER_MODE="$(plist_value "${DEVICES_JSON}" "result.devices.${SELECTED_INDEX}.deviceProperties.developerModeStatus")"
DDI_AVAILABLE="$(plist_value "${DEVICES_JSON}" "result.devices.${SELECTED_INDEX}.deviceProperties.ddiServicesAvailable")"
DEVICE_ID="${DEVICE_UDID:-${DEVICE_IDENTIFIER}}"

[[ -n "${DEVICE_ID}" ]] || fail "the selected device has no usable identifier"
[[ "${DEVELOPER_MODE}" == "enabled" ]] || fail "Developer Mode is not enabled on '${DEVICE_NAME}'"
[[ "${DDI_AVAILABLE}" == "true" ]] || fail "developer services are unavailable on '${DEVICE_NAME}'; unlock it and reconnect"

# Select a development identity, never a distribution identity. Keep its SHA-1
# alongside the display name so automatic profile selection can prove that the
# certificate embedded in the profile is the certificate we will sign with.
DEVELOPMENT_IDENTITIES="$(security find-identity -v -p codesigning 2>/dev/null | \
    awk '/"Apple Development:/ {
        hash = $2
        name = $0
        sub(/^[^"]*"/, "", name)
        sub(/"[^"]*$/, "", name)
        print hash "\t" name
    }')"
if [[ -z "${CODESIGN_KEY:-}" ]]; then
    IDENTITY_COUNT="$(printf '%s\n' "${DEVELOPMENT_IDENTITIES}" | awk 'NF { count++ } END { print count + 0 }')"
    if [[ "${IDENTITY_COUNT}" -ne 1 ]]; then
        echo "Available Apple Development identities:" >&2
        printf '%s\n' "${DEVELOPMENT_IDENTITIES:-  (none)}" >&2
        fail "found ${IDENTITY_COUNT}; set CODESIGN_KEY to the identity to use"
    fi
    CODESIGN_IDENTITY_HASH="$(printf '%s\n' "${DEVELOPMENT_IDENTITIES}" | awk -F '\t' 'NF { print $1; exit }')"
    CODESIGN_KEY="$(printf '%s\n' "${DEVELOPMENT_IDENTITIES}" | awk -F '\t' 'NF { print $2; exit }')"
else
    CODESIGN_IDENTITY_HASH="$(printf '%s\n' "${DEVELOPMENT_IDENTITIES}" | \
        awk -F '\t' -v requested="${CODESIGN_KEY}" '$1 == requested || $2 == requested { print $1; exit }')"
    [[ -n "${CODESIGN_IDENTITY_HASH}" ]] || fail "CODESIGN_KEY does not identify a valid Apple Development certificate"
fi

# Resolve profiles ourselves instead of allowing one global CodesignProvision to leak
# into the nested .appex. A containing app and an extension are two independently
# signed bundles and Apple requires an exact profile for each identifier. When sharing
# is enabled, BOTH profiles must also grant the exact App Group used for the durable
# inbox. This is checked for caller-supplied profiles too; an override is not a bypass.
PROFILE_PLIST="${TEMP_DIR}/profile.plist"
PROFILE_CERTIFICATE="${TEMP_DIR}/profile-certificate.der"
NOW_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
shopt -s nullglob
PROFILE_FILES=(
    "${HOME}/Library/MobileDevice/Provisioning Profiles"/*.mobileprovision
    "${HOME}/Library/MobileDevice/Provisioning Profiles"/*.provisionprofile
    "${HOME}/Library/Developer/Xcode/UserData/Provisioning Profiles"/*.mobileprovision
    "${HOME}/Library/Developer/Xcode/UserData/Provisioning Profiles"/*.provisionprofile
)
shopt -u nullglob

profile_contains_signing_certificate() { # decoded-profile
    local profile="$1" count index hash
    count="$(plist_value "${profile}" DeveloperCertificates)"
    [[ "${count}" =~ ^[0-9]+$ ]] || return 1
    index=0
    while [[ "${index}" -lt "${count}" ]]; do
        if plutil -extract "DeveloperCertificates.${index}" raw -o - "${profile}" 2>/dev/null | \
            base64 -D > "${PROFILE_CERTIFICATE}" 2>/dev/null; then
            hash="$(shasum -a 1 "${PROFILE_CERTIFICATE}" | awk '{ print toupper($1) }')"
            [[ "${hash}" == "${CODESIGN_IDENTITY_HASH}" ]] && return 0
        fi
        index=$((index + 1))
    done
    return 1
}

profile_contains_device() { # decoded-profile
    local profile="$1" count index
    count="$(plist_value "${profile}" ProvisionedDevices)"
    [[ "${count}" =~ ^[0-9]+$ ]] || return 1
    index=0
    while [[ "${index}" -lt "${count}" ]]; do
        [[ "$(plist_value "${profile}" "ProvisionedDevices.${index}")" == "${DEVICE_UDID}" ]] && return 0
        index=$((index + 1))
    done
    return 1
}

profile_grants_boolean_entitlements() { # decoded-profile required-entitlements-plist
    local profile="$1" required="$2" entitlement
    while IFS= read -r entitlement; do
        [[ -n "${entitlement}" ]] || continue
        [[ "$(/usr/libexec/PlistBuddy -c "Print :Entitlements:${entitlement}" "${profile}" 2>/dev/null || true)" == "true" ]] \
            || return 1
    done < <(/usr/libexec/PlistBuddy -c Print "${required}" | \
        sed -nE 's/^[[:space:]]*([^ =]+)[[:space:]]*=[[:space:]]*true$/\1/p')
    return 0
}

# Sets SELECTED_PROFILE_UUID and SELECTED_PROFILE_DESCRIPTION. The optional requested
# value is a profile name or UUID; required_group is empty for an app-only build.
select_profile() { # label exact-bundle-id requested-profile required-group required-boolean-entitlements
    local label="$1" bundle_id="$2" requested="$3" required_group="$4" required_booleans="$5"
    local profile_file profile_app_id profile_bundle_id profile_uuid profile_name profile_expiry
    local best_uuid="" best_name="" best_expiry=""

    for profile_file in "${PROFILE_FILES[@]}"; do
        security cms -D -i "${profile_file}" > "${PROFILE_PLIST}" 2>/dev/null || continue
        profile_uuid="$(plist_value "${PROFILE_PLIST}" UUID)"
        profile_name="$(plist_value "${PROFILE_PLIST}" Name)"
        if [[ -n "${requested}" && "${requested}" != "${profile_uuid}" && "${requested}" != "${profile_name}" ]]; then
            continue
        fi

        profile_app_id="$(plist_value "${PROFILE_PLIST}" Entitlements.application-identifier)"
        profile_bundle_id="${profile_app_id#*.}"
        [[ "${profile_bundle_id}" == "${bundle_id}" ]] || continue
        [[ "$(plist_value "${PROFILE_PLIST}" Entitlements.get-task-allow)" == "true" ]] || continue
        profile_contains_signing_certificate "${PROFILE_PLIST}" || continue
        profile_contains_device "${PROFILE_PLIST}" || continue

        profile_expiry="$(plist_value "${PROFILE_PLIST}" ExpirationDate)"
        [[ -n "${profile_expiry}" && "${profile_expiry}" > "${NOW_UTC}" ]] || continue
        if [[ -n "${required_group}" ]]; then
            plist_array_contains "${PROFILE_PLIST}" \
                "Entitlements:com.apple.security.application-groups" "${required_group}" || continue
        fi
        if [[ -n "${required_booleans}" ]]; then
            profile_grants_boolean_entitlements "${PROFILE_PLIST}" "${required_booleans}" || continue
        fi

        [[ -n "${profile_uuid}" ]] || continue
        if [[ -z "${best_uuid}" || "${profile_expiry}" > "${best_expiry}" ]]; then
            best_uuid="${profile_uuid}"
            best_name="${profile_name}"
            best_expiry="${profile_expiry}"
        fi
    done

    if [[ -z "${best_uuid}" ]]; then
        if [[ -n "${requested}" ]]; then
            fail "${label} profile '${requested}' is not an unexpired development profile for exact bundle id ${bundle_id}, this signing certificate, and ${DEVICE_NAME}${required_group:+ with App Group ${required_group}}"
        fi
        fail "no unexpired ${label} development profile for exact bundle id ${bundle_id} contains ${DEVICE_NAME}${required_group:+ and App Group ${required_group}}; refresh signing in Xcode or the Developer portal"
    fi

    SELECTED_PROFILE_UUID="${best_uuid}"
    SELECTED_PROFILE_DESCRIPTION="${best_name} (${best_uuid}, expires ${best_expiry})"
}

MAIN_REQUIRED_GROUP=""
if [[ "${TENSORAGENT_SHARE_EXTENSION}" == "true" ]]; then
    MAIN_REQUIRED_GROUP="${APP_GROUP}"
fi
REQUESTED_SHARE_PROFILE="${CODESIGN_SHARE_PROVISION:-}"
select_profile "containing-app" "${BUNDLE_ID}" "${CODESIGN_PROVISION:-}" \
    "${MAIN_REQUIRED_GROUP}" "${ENTITLEMENTS}"
CODESIGN_PROVISION="${SELECTED_PROFILE_UUID}"
MAIN_PROFILE_DESCRIPTION="${SELECTED_PROFILE_DESCRIPTION}"

CODESIGN_SHARE_PROVISION=""
SHARE_PROFILE_DESCRIPTION="disabled"
if [[ "${TENSORAGENT_SHARE_EXTENSION}" == "true" ]]; then
    select_profile "share-extension" "${SHARE_BUNDLE_ID}" "${REQUESTED_SHARE_PROFILE}" \
        "${APP_GROUP}" ""
    CODESIGN_SHARE_PROVISION="${SELECTED_PROFILE_UUID}"
    SHARE_PROFILE_DESCRIPTION="${SELECTED_PROFILE_DESCRIPTION}"
    [[ "${CODESIGN_SHARE_PROVISION}" != "${CODESIGN_PROVISION}" ]] \
        || fail "the containing app and share extension resolved to the same profile; each bundle needs its own exact profile"
fi

PYTHON_FRAMEWORK="${REPO_ROOT}/TensorAgent/python-runtime/device/Frameworks/Python.framework"
PYTHON_STDLIB="${REPO_ROOT}/TensorAgent/python-runtime/device/python"
if [[ ! -d "${PYTHON_FRAMEWORK}" || ! -d "${PYTHON_STDLIB}" ]]; then
    fail "the device Python runtime is not staged; run TensorAgent/scripts/prepare-python.sh device"
fi

echo "==> Target: ${DEVICE_NAME} (${DEVICE_ID})"
echo "==> Signing identity: ${CODESIGN_KEY}"
echo "==> Containing-app profile: ${MAIN_PROFILE_DESCRIPTION}"
echo "==> Share-extension profile: ${SHARE_PROFILE_DESCRIPTION}"
echo "==> Building a fresh TensorAgent ${CONFIGURATION} bundle"
CONFIGURATION="${CONFIGURATION}" \
CODESIGN_KEY="${CODESIGN_KEY}" \
CODESIGN_PROVISION="${CODESIGN_PROVISION}" \
CODESIGN_SHARE_PROVISION="${CODESIGN_SHARE_PROVISION}" \
TENSORAGENT_SHARE_EXTENSION="${TENSORAGENT_SHARE_EXTENSION}" \
CLEAN=1 \
NO_INCREMENTAL=1 \
SKIP_SIGNING=0 \
TENSORAGENT_REBUILD_XCFRAMEWORK="${TENSORAGENT_REBUILD_XCFRAMEWORK}" \
    bash "${SCRIPT_DIR}/build-device.sh"

[[ -d "${APP}" ]] || fail "build completed without producing ${APP}"
codesign --verify --deep --strict --verbose=2 "${APP}"
ACTUAL_BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "${APP}/Info.plist" 2>/dev/null || true)"
[[ "${ACTUAL_BUNDLE_ID}" == "${BUNDLE_ID}" ]] || fail "built bundle identifier is '${ACTUAL_BUNDLE_ID}', expected '${BUNDLE_ID}'"

# Check what the SDK actually embedded and signed, not just the inputs handed to
# MSBuild. A nested project can quietly inherit the containing app's profile, and an
# App Group present in a portal profile but absent from the code signature still makes
# containerURLForSecurityApplicationGroupIdentifier return nil at runtime.
validate_built_bundle() { # label bundle exact-bundle-id selected-profile-uuid group-mode
    local label="$1" bundle="$2" bundle_id="$3" selected_uuid="$4" group_mode="$5"
    local embedded="${bundle}/embedded.mobileprovision"
    local decoded="${TEMP_DIR}/${label}-embedded.plist"
    local signed="${TEMP_DIR}/${label}-signed-entitlements.plist"
    local actual_id embedded_uuid profile_app_id profile_bundle_id signed_app_id signed_bundle_id

    [[ -f "${bundle}/Info.plist" ]] || fail "${label} bundle has no Info.plist"
    actual_id="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "${bundle}/Info.plist" 2>/dev/null || true)"
    [[ "${actual_id}" == "${bundle_id}" ]] \
        || fail "${label} bundle identifier is '${actual_id}', expected '${bundle_id}'"

    [[ -f "${embedded}" ]] || fail "${label} bundle has no embedded provisioning profile"
    security cms -D -i "${embedded}" > "${decoded}" 2>/dev/null \
        || fail "${label} embedded provisioning profile could not be decoded"
    embedded_uuid="$(plist_value "${decoded}" UUID)"
    [[ "${embedded_uuid}" == "${selected_uuid}" ]] \
        || fail "${label} embedded profile is ${embedded_uuid:-unknown}, expected ${selected_uuid}"
    profile_app_id="$(plist_value "${decoded}" Entitlements.application-identifier)"
    profile_bundle_id="${profile_app_id#*.}"
    [[ "${profile_bundle_id}" == "${bundle_id}" ]] \
        || fail "${label} embedded profile is for '${profile_bundle_id}', expected exact id '${bundle_id}'"

    codesign -d --entitlements :- "${bundle}" > "${signed}" 2>/dev/null \
        || fail "${label} signed entitlements could not be read"
    signed_app_id="$(plist_value "${signed}" application-identifier)"
    signed_bundle_id="${signed_app_id#*.}"
    [[ "${signed_bundle_id}" == "${bundle_id}" ]] \
        || fail "${label} signature application-identifier is '${signed_app_id}', expected exact id '${bundle_id}'"

    if [[ "${group_mode}" == "required" ]]; then
        plist_array_contains "${decoded}" "Entitlements:com.apple.security.application-groups" "${APP_GROUP}" \
            || fail "${label} embedded profile does not grant App Group ${APP_GROUP}"
        plist_array_contains "${signed}" "com.apple.security.application-groups" "${APP_GROUP}" \
            || fail "${label} code signature does not request App Group ${APP_GROUP}"
    elif plist_array_contains "${signed}" "com.apple.security.application-groups" "${APP_GROUP}"; then
        fail "${label} code signature still requests ${APP_GROUP} although the share extension is disabled"
    fi
}

MAIN_GROUP_MODE="forbidden"
[[ "${TENSORAGENT_SHARE_EXTENSION}" == "true" ]] && MAIN_GROUP_MODE="required"
validate_built_bundle "main" "${APP}" "${BUNDLE_ID}" "${CODESIGN_PROVISION}" "${MAIN_GROUP_MODE}"

# The main app's two device-only memory entitlements are independent of sharing and
# must survive either mode. They were already checked against the selected profile;
# repeat the check against the final signature.
MAIN_SIGNED_ENTITLEMENTS="${TEMP_DIR}/main-signed-entitlements.plist"
while IFS= read -r ENTITLEMENT; do
    [[ -n "${ENTITLEMENT}" ]] || continue
    [[ "$(/usr/libexec/PlistBuddy -c "Print :${ENTITLEMENT}" "${MAIN_SIGNED_ENTITLEMENTS}" 2>/dev/null || true)" == "true" ]] \
        || fail "containing-app signature is missing required entitlement ${ENTITLEMENT}"
done < <(/usr/libexec/PlistBuddy -c Print "${ENTITLEMENTS}" | \
    sed -nE 's/^[[:space:]]*([^ =]+)[[:space:]]*=[[:space:]]*true$/\1/p')

SHARE_APPEX=""
if [[ -d "${APP}/PlugIns" ]]; then
    while IFS= read -r CANDIDATE_APPEX; do
        CANDIDATE_BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "${CANDIDATE_APPEX}/Info.plist" 2>/dev/null || true)"
        if [[ "${CANDIDATE_BUNDLE_ID}" == "${SHARE_BUNDLE_ID}" ]]; then
            [[ -z "${SHARE_APPEX}" ]] || fail "the app contains more than one ${SHARE_BUNDLE_ID} extension"
            SHARE_APPEX="${CANDIDATE_APPEX}"
        fi
    done < <(find "${APP}/PlugIns" -mindepth 1 -maxdepth 1 -type d -name '*.appex' -print)
fi

if [[ "${TENSORAGENT_SHARE_EXTENSION}" == "true" ]]; then
    [[ -n "${SHARE_APPEX}" ]] || fail "share extension ${SHARE_BUNDLE_ID} was enabled but is absent from the app bundle"
    validate_built_bundle "share" "${SHARE_APPEX}" "${SHARE_BUNDLE_ID}" \
        "${CODESIGN_SHARE_PROVISION}" "required"
    [[ -f "${SHARE_APPEX}/PrivacyInfo.xcprivacy" ]] \
        || fail "share extension is missing PrivacyInfo.xcprivacy at its bundle root"

    MAIN_SHORT_VERSION="$(plist_value "${APP}/Info.plist" CFBundleShortVersionString)"
    MAIN_BUILD_VERSION="$(plist_value "${APP}/Info.plist" CFBundleVersion)"
    SHARE_SHORT_VERSION="$(plist_value "${SHARE_APPEX}/Info.plist" CFBundleShortVersionString)"
    SHARE_BUILD_VERSION="$(plist_value "${SHARE_APPEX}/Info.plist" CFBundleVersion)"
    [[ "${SHARE_SHORT_VERSION}" == "${MAIN_SHORT_VERSION}" && "${SHARE_BUILD_VERSION}" == "${MAIN_BUILD_VERSION}" ]] \
        || fail "share extension version ${SHARE_SHORT_VERSION} (${SHARE_BUILD_VERSION}) does not match app ${MAIN_SHORT_VERSION} (${MAIN_BUILD_VERSION})"
elif [[ -n "${SHARE_APPEX}" ]]; then
    fail "share extension was disabled but ${SHARE_BUNDLE_ID} is still embedded"
fi

APP_EXECUTABLE="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "${APP}/Info.plist" 2>/dev/null || true)"
[[ -n "${APP_EXECUTABLE}" && -f "${APP}/${APP_EXECUTABLE}" ]] || fail "could not locate the built app executable"
nm -gU "${APP}/${APP_EXECUTABLE}" > "${TEMP_DIR}/symbols.txt"
grep -q ' _TSGgml_IsMetalAvailable$' "${TEMP_DIR}/symbols.txt" \
    || fail "${CONFIGURATION} bundle is missing TensorAgent's GGML exports"

echo "==> Installing TensorAgent on ${DEVICE_NAME} (existing app data is preserved)"
xcrun devicectl --timeout "${DEVICECTL_TIMEOUT}" device install app \
    --device "${DEVICE_ID}" "${APP}"

if [[ "${SKIP_LAUNCH:-0}" != "1" ]]; then
    echo "==> Launching ${BUNDLE_ID}"
    xcrun devicectl --timeout 60 device process launch \
        --device "${DEVICE_ID}" --terminate-existing "${BUNDLE_ID}"
fi

echo "==> TensorAgent ${CONFIGURATION} deployed successfully to ${DEVICE_NAME}"
