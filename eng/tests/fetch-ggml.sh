#!/usr/bin/env bash
# Exercise the real fetch helper against a disposable local Git repository.
# No network, model files, or existing ExternalProjects checkout are touched.
# Run from any directory: bash eng/tests/fetch-ggml.sh [--powershell]
set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FETCH_NAME=fetch-ggml.sh
FETCH_RUNNER=(bash)
if [[ "${1:-}" == --powershell ]]; then
    FETCH_NAME=fetch-ggml.ps1
    FETCH_RUNNER=(pwsh -NoLogo -NoProfile -File)
elif [[ $# != 0 ]]; then
    echo 'Usage: fetch-ggml.sh [--powershell]' >&2
    exit 2
fi
FETCH_SCRIPT="${TEST_DIR}/../${FETCH_NAME}"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/tensorsharp-fetch-ggml.XXXXXX")"
trap 'rm -rf "${TEST_ROOT}"' EXIT

# Keep local user signing, hooks, line-ending settings, and credentials out of
# the fixture. The protocol allowlist makes an accidental network URL fail.
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_ALLOW_PROTOCOL=file
export GIT_TERMINAL_PROMPT=0

UPSTREAM="${TEST_ROOT}/upstream"
WORKSPACE="${TEST_ROOT}/workspace with spaces"
mkdir -p "${UPSTREAM}" "${WORKSPACE}/eng/ggml-patches"
git -C "${UPSTREAM}" init -q
git -C "${UPSTREAM}" symbolic-ref HEAD refs/heads/master
git -C "${UPSTREAM}" config user.name 'Fetch regression test'
git -C "${UPSTREAM}" config user.email 'fetch-test@example.invalid'
printf 'upstream version one\n' > "${UPSTREAM}/source.txt"
git -C "${UPSTREAM}" add source.txt
git -C "${UPSTREAM}" commit -qm 'Initial upstream'
FIRST_COMMIT="$(git -C "${UPSTREAM}" rev-parse HEAD)"
git -C "${UPSTREAM}" tag fixture-v1
cp "${FETCH_SCRIPT}" "${WORKSPACE}/eng/${FETCH_NAME}"

# A stale local patch left by an older TensorSharp checkout must have no effect.
# This patch is valid, so reintroducing automatic patch application would dirty
# the fixture and fail the first clean-checkout assertion.
cat > "${WORKSPACE}/eng/ggml-patches/legacy.patch" <<'PATCH'
diff --git a/source.txt b/source.txt
--- a/source.txt
+++ b/source.txt
@@ -1 +1 @@
-upstream version one
+unexpected local modification
PATCH

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

run_fetch() {
    local workspace="$1" ref="$2" no_update="${3:-}" url="${4:-${UPSTREAM}}"
    if ! TENSORSHARP_GGML_GIT_URL="${url}" \
         TENSORSHARP_GGML_GIT_REF="${ref}" \
         TENSORSHARP_GGML_NO_UPDATE="${no_update}" \
         "${FETCH_RUNNER[@]}" "${workspace}/eng/${FETCH_NAME}" > "${TEST_ROOT}/fetch.log" 2>&1; then
        cat "${TEST_ROOT}/fetch.log" >&2
        fail "fetch failed for ref ${ref} (no-update=${no_update})"
    fi
}

assert_clean_checkout() {
    local workspace="$1" expected="$2" label="$3"
    local checkout="${workspace}/ExternalProjects/ggml"
    [[ "$(git -C "${checkout}" rev-parse HEAD)" == "${expected}" ]] \
        || fail "${label}: unexpected commit"
    [[ -z "$(git -C "${checkout}" status --porcelain --untracked-files=all)" ]] \
        || fail "${label}: upstream sources were modified"
    git -C "${checkout}" diff --quiet "${expected}" -- \
        || fail "${label}: checkout differs from upstream"
    echo "PASS: ${label}"
}

# Initial clone and branch update, including a left-over legacy patch directory.
run_fetch "${WORKSPACE}" master
assert_clean_checkout "${WORKSPACE}" "${FIRST_COMMIT}" 'branch clone stays clean'
printf 'upstream version two\n' > "${UPSTREAM}/source.txt"
git -C "${UPSTREAM}" commit -qam 'Advance upstream'
SECOND_COMMIT="$(git -C "${UPSTREAM}" rev-parse HEAD)"
run_fetch "${WORKSPACE}" master
assert_clean_checkout "${WORKSPACE}" "${SECOND_COMMIT}" 'branch update stays clean'

# NO_UPDATE must neither fetch a new commit nor rewrite origin, even with an
# unavailable requested URL. Exercise the documented spellings.
ORIGINAL_ORIGIN="$(git -C "${WORKSPACE}/ExternalProjects/ggml" remote get-url origin)"
for truthy in 1 ON true; do
    run_fetch "${WORKSPACE}" fixture-v1 "${truthy}" "${TEST_ROOT}/missing-origin"
    assert_clean_checkout "${WORKSPACE}" "${SECOND_COMMIT}" "no-update ${truthy} preserves checkout"
    [[ "$(git -C "${WORKSPACE}/ExternalProjects/ggml" remote get-url origin)" == "${ORIGINAL_ORIGIN}" ]] \
        || fail "no-update ${truthy}: origin was rewritten"
done

# A model-validation VM may receive exact sources without .git. NO_UPDATE must
# preserve those bytes and never try the deliberately unavailable origin.
SOURCE_WORKSPACE="${TEST_ROOT}/source copy"
SOURCE_DIR="${SOURCE_WORKSPACE}/ExternalProjects/ggml"
mkdir -p "${SOURCE_WORKSPACE}/eng" "${SOURCE_DIR}/src" "${SOURCE_DIR}/include"
cp "${FETCH_SCRIPT}" "${SOURCE_WORKSPACE}/eng/${FETCH_NAME}"
printf 'pinned build source\n' > "${SOURCE_DIR}/CMakeLists.txt"
printf 'pinned targets\n' > "${SOURCE_DIR}/src/CMakeLists.txt"
printf 'pinned header\n' > "${SOURCE_DIR}/include/ggml.h"
cp -R "${SOURCE_DIR}" "${TEST_ROOT}/expected-source-copy"
for truthy in 1 ON true; do
    run_fetch "${SOURCE_WORKSPACE}" master "${truthy}" "${TEST_ROOT}/missing-origin"
    diff -r "${TEST_ROOT}/expected-source-copy" "${SOURCE_DIR}" \
        || fail "no-update ${truthy}: source copy changed"
    [[ ! -e "${SOURCE_DIR}/.git" ]] || fail 'source copy was replaced by a checkout'
    echo "PASS: no-update ${truthy} preserves complete source copy without Git metadata"
done

# Merely having a directory (or only some build inputs) must not masquerade as
# a complete source copy. Every required marker is checked independently.
for missing in CMakeLists.txt src/CMakeLists.txt include/ggml.h; do
    INCOMPLETE_WORKSPACE="${TEST_ROOT}/incomplete-${missing//\//-}"
    mkdir -p "${INCOMPLETE_WORKSPACE}/eng" "${INCOMPLETE_WORKSPACE}/ExternalProjects"
    cp "${FETCH_SCRIPT}" "${INCOMPLETE_WORKSPACE}/eng/${FETCH_NAME}"
    cp -R "${TEST_ROOT}/expected-source-copy" "${INCOMPLETE_WORKSPACE}/ExternalProjects/ggml"
    rm "${INCOMPLETE_WORKSPACE}/ExternalProjects/ggml/${missing}"
    run_fetch "${INCOMPLETE_WORKSPACE}" master true
    assert_clean_checkout "${INCOMPLETE_WORKSPACE}" "${SECOND_COMMIT}" "incomplete source copy missing ${missing} fetches normally"
done

# Offline fallback must preserve the existing checkout, then recover normally
# when the upstream URL becomes available again.
run_fetch "${WORKSPACE}" master '' "${TEST_ROOT}/missing-origin"
assert_clean_checkout "${WORKSPACE}" "${SECOND_COMMIT}" 'offline fallback stays clean'
[[ "$(cat "${TEST_ROOT}/fetch.log")" == *'could not fetch'* ]] \
    || fail 'offline fallback did not report the fetch failure'
run_fetch "${WORKSPACE}" fixture-v1
assert_clean_checkout "${WORKSPACE}" "${FIRST_COMMIT}" 'custom tag update stays clean'
run_fetch "${WORKSPACE}" "${SECOND_COMMIT}"
assert_clean_checkout "${WORKSPACE}" "${SECOND_COMMIT}" 'custom commit update stays clean'

# Preserve deliberate user edits on paths that promise to use existing sources.
printf 'local user edit\n' >> "${WORKSPACE}/ExternalProjects/ggml/source.txt"
cp "${WORKSPACE}/ExternalProjects/ggml/source.txt" "${TEST_ROOT}/expected-dirty.txt"
run_fetch "${WORKSPACE}" master true
cmp -s "${TEST_ROOT}/expected-dirty.txt" "${WORKSPACE}/ExternalProjects/ggml/source.txt" \
    || fail 'no-update changed local edits'
run_fetch "${WORKSPACE}" master '' "${TEST_ROOT}/missing-origin"
cmp -s "${TEST_ROOT}/expected-dirty.txt" "${WORKSPACE}/ExternalProjects/ggml/source.txt" \
    || fail 'offline fallback changed local edits'
echo 'PASS: no-update and offline fallback preserve local edits'

# Explicit commit IDs take the clone fallback (git clone --branch accepts tags
# and branches only). Each gets a fresh workspace, including partial-clone debris.
for ref in fixture-v1 "${FIRST_COMMIT}"; do
    FRESH_WORKSPACE="${TEST_ROOT}/fresh-${ref}"
    mkdir -p "${FRESH_WORKSPACE}/eng" "${FRESH_WORKSPACE}/ExternalProjects/ggml"
    cp "${FETCH_SCRIPT}" "${FRESH_WORKSPACE}/eng/${FETCH_NAME}"
    printf 'partial clone\n' > "${FRESH_WORKSPACE}/ExternalProjects/ggml/incomplete"
    run_fetch "${FRESH_WORKSPACE}" "${ref}"
    assert_clean_checkout "${FRESH_WORKSPACE}" "${FIRST_COMMIT}" "fresh custom ref ${ref} stays clean"
done

[[ -z "$(git -C "${UPSTREAM}" status --porcelain --untracked-files=all)" ]] \
    || fail 'fixture upstream was modified'
[[ "$(git -C "${UPSTREAM}" rev-parse HEAD)" == "${SECOND_COMMIT}" ]] \
    || fail 'fixture upstream HEAD changed'
echo 'PASS: fetch-ggml regression checks completed without network access'
