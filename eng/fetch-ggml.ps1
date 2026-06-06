# Clone the upstream ggml-org/ggml repository into ExternalProjects/ggml when
# the sources are missing.
#
# TensorSharp builds ggml from source: both the GGML native ops library
# (TensorSharp.GGML.Native/CMakeLists.txt) and the CUDA PTX kernels
# (TensorSharp.Backends.Cuda/native/kernels/tensorsharp_kernels.cu) consume the
# sources at ExternalProjects/ggml. The directory is not committed; it is fetched
# here on first use, then reused for faster and offline-friendly rebuilds.
#
# Environment overrides:
#   TENSORSHARP_GGML_GIT_URL   git URL                (default: ggml-org/ggml)
#   TENSORSHARP_GGML_GIT_REF   branch/tag/commit      (default: master, the ggml default branch)
#   TENSORSHARP_GGML_UPDATE    if set to 1/ON/true and a checkout already exists,
#                              fetch and reset it to the requested ref.
#   TENSORSHARP_GGML_NO_UPDATE legacy override; if truthy, always use what is on disk.
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$GgmlDir = Join-Path $RepoRoot "ExternalProjects/ggml"
$GgmlRequiredHeader = Join-Path $GgmlDir "src\ggml-common.h"

$GitUrl = if ([string]::IsNullOrWhiteSpace($env:TENSORSHARP_GGML_GIT_URL)) { "https://github.com/ggml-org/ggml.git" } else { $env:TENSORSHARP_GGML_GIT_URL }
$GitRef = if ([string]::IsNullOrWhiteSpace($env:TENSORSHARP_GGML_GIT_REF)) { "master" } else { $env:TENSORSHARP_GGML_GIT_REF }

function Test-Truthy([string] $Value) {
    return $Value -match '^(1|ON|on|On|TRUE|true|True|YES|yes|Yes)$'
}

if (Test-Path (Join-Path $GgmlDir ".git")) {
    if (Test-Truthy $env:TENSORSHARP_GGML_NO_UPDATE) {
        Write-Host "ggml: TENSORSHARP_GGML_NO_UPDATE set; using existing checkout at $GgmlDir"
        exit 0
    }

    if ((Test-Path $GgmlRequiredHeader) -and -not (Test-Truthy $env:TENSORSHARP_GGML_UPDATE)) {
        Write-Host "ggml: using existing checkout at $GgmlDir (set TENSORSHARP_GGML_UPDATE=1 to fetch updates)"
        exit 0
    }

    if (Test-Path $GgmlRequiredHeader) {
        Write-Host "ggml: updating existing checkout to $GitRef ($GitUrl)"
    }
    else {
        Write-Host "ggml: existing checkout is missing src/ggml-common.h; fetching $GitRef ($GitUrl)"
    }

    git -C $GgmlDir remote set-url origin $GitUrl 2>$null
    git -C $GgmlDir fetch --depth 1 origin $GitRef 2>$null
    if ($LASTEXITCODE -eq 0) {
        git -C $GgmlDir reset --hard FETCH_HEAD
        if ($LASTEXITCODE -ne 0) {
            if (Test-Path $GgmlRequiredHeader) {
                Write-Warning "ggml: fetched $GitRef, but could not reset checkout; using existing sources"
                exit 0
            }

            throw "git reset failed and no usable ggml sources exist at $GgmlDir"
        }
        $sha = (git -C $GgmlDir rev-parse --short HEAD).Trim()
        Write-Host "ggml: now at $sha"
    }
    else {
        if (Test-Path $GgmlRequiredHeader) {
            Write-Warning "ggml: could not fetch $GitRef (offline?); using existing checkout"
        }
        else {
            throw "could not fetch $GitRef, and no usable ggml sources exist at $GgmlDir"
        }
    }
    exit 0
}

if ((Test-Path $GgmlRequiredHeader) -and -not (Test-Truthy $env:TENSORSHARP_GGML_UPDATE)) {
    Write-Host "ggml: using existing source tree at $GgmlDir (not a git checkout)"
    exit 0
}

# No checkout yet: clear a partial/empty directory left by a failed clone.
if (Test-Path $GgmlDir) {
    Remove-Item -Recurse -Force $GgmlDir
}

Write-Host "ggml: cloning $GitUrl ($GitRef) into $GgmlDir"
git clone --depth 1 --branch $GitRef $GitUrl $GgmlDir 2>$null
if ($LASTEXITCODE -ne 0) {
    # --branch only accepts a branch or tag; fall back to fetching an explicit
    # commit ref shallowly.
    if (Test-Path $GgmlDir) { Remove-Item -Recurse -Force $GgmlDir }
    git init -q $GgmlDir
    if ($LASTEXITCODE -ne 0) { throw "git init failed" }
    git -C $GgmlDir remote add origin $GitUrl
    git -C $GgmlDir fetch --depth 1 origin $GitRef
    if ($LASTEXITCODE -ne 0) { throw "git fetch '$GitRef' failed" }
    git -C $GgmlDir checkout -q FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed" }
}
$sha = (git -C $GgmlDir rev-parse --short HEAD).Trim()
Write-Host "ggml: cloned at $sha"
