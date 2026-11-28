# Clone (or update) the upstream ggml-org/ggml repository into ExternalProjects/ggml.
#
# TensorSharp builds ggml from source: both the GGML native ops library
# (TensorSharp.GGML.Native/CMakeLists.txt) and the CUDA PTX kernels
# (TensorSharp.Backends.Cuda/native/kernels/tensorsharp_kernels.cu) consume the
# sources at ExternalProjects/ggml. The directory is not committed; it is fetched
# here at build time from upstream ggml's latest release (its newest vX.Y.Z tag),
# not from its moving master branch.
#
# Environment overrides:
#   TENSORSHARP_GGML_GIT_URL   git URL                (default: ggml-org/ggml)
#   TENSORSHARP_GGML_GIT_REF   branch/tag/commit, or latest-release (default): the
#                              highest stable vX.Y.Z tag at the URL, looked up with
#                              git ls-remote (master follows the default branch)
#   TENSORSHARP_GGML_NO_UPDATE if set to 1/ON/true and a checkout or complete
#                              source copy exists, use what is on disk.
# Fetching and building leave upstream sources unchanged. Model-specific
# behavior and precision kernels live in TensorSharp.GGML.Native.
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$ExternalProjectsDir = Join-Path $RepoRoot "ExternalProjects"
$GgmlDir = Join-Path $ExternalProjectsDir "ggml"

$GitUrl = if ([string]::IsNullOrWhiteSpace($env:TENSORSHARP_GGML_GIT_URL)) { "https://github.com/ggml-org/ggml.git" } else { $env:TENSORSHARP_GGML_GIT_URL }
$GitRef = if ([string]::IsNullOrWhiteSpace($env:TENSORSHARP_GGML_GIT_REF)) { "latest-release" } else { $env:TENSORSHARP_GGML_GIT_REF }

New-Item -ItemType Directory -Force -Path $ExternalProjectsDir | Out-Null
$Sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
    $HashBytes = $Sha256.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($RepoRoot.ToLowerInvariant()))
}
finally {
    $Sha256.Dispose()
}
$HashText = -join ($HashBytes[0..7] | ForEach-Object { $_.ToString("x2") })
$FetchMutex = [System.Threading.Mutex]::new($false, "TensorSharpGgmlFetch_$HashText")
$HasFetchMutex = $false

try {
    $HasFetchMutex = $FetchMutex.WaitOne([TimeSpan]::FromMinutes(10))
    if (-not $HasFetchMutex) {
        throw "Timed out waiting for ggml fetch lock."
    }

function Test-Truthy([string] $Value) {
    return $Value -match '^(1|ON|on|On|TRUE|true|True|YES|yes|Yes)$'
}

# Pinned source archives and rsync copies may intentionally omit Git metadata.
# Keep valid sources before partial-clone cleanup when no-update is requested.
if ((Test-Truthy $env:TENSORSHARP_GGML_NO_UPDATE) -and
    (Test-Path (Join-Path $GgmlDir "CMakeLists.txt") -PathType Leaf) -and
    (Test-Path (Join-Path $GgmlDir "src/CMakeLists.txt") -PathType Leaf) -and
    (Test-Path (Join-Path $GgmlDir "include/ggml.h") -PathType Leaf)) {
    Write-Host "ggml: TENSORSHARP_GGML_NO_UPDATE set; using existing sources at $GgmlDir"
    exit 0
}

$HasCheckout = Test-Path (Join-Path $GgmlDir ".git")
if ($HasCheckout -and (Test-Truthy $env:TENSORSHARP_GGML_NO_UPDATE)) {
    Write-Host "ggml: TENSORSHARP_GGML_NO_UPDATE set; using existing checkout at $GgmlDir"
    exit 0
}

# The newest stable release tag (vMAJOR.MINOR.PATCH, no pre-release suffix) at the
# URL, compared numerically (v1.10.0 after v1.9.0); $null when the remote cannot
# be listed (offline) or has none. git's stderr is not redirected (see below).
function Get-LatestRelease {
    $lines = @(git ls-remote --tags --refs $GitUrl 'refs/tags/v*')
    if ($LASTEXITCODE -ne 0) { return $null }
    $tags = foreach ($line in $lines) {
        $name = ($line -split '\s+')[1] -replace '^refs/tags/', ''
        if ($name -match '^v(\d+)\.(\d+)\.(\d+)$') {
            [pscustomobject]@{ Tag = $name; Version = [version]::new([int]$Matches[1], [int]$Matches[2], [int]$Matches[3]) }
        }
    }
    $latest = $tags | Sort-Object Version | Select-Object -Last 1
    if ($null -eq $latest) { return $null }
    return $latest.Tag
}

if ($GitRef -eq "latest-release") {
    $Release = Get-LatestRelease
    if ($Release) {
        Write-Host "ggml: latest release at $GitUrl is $Release"
        $GitRef = $Release
    }
    elseif ($HasCheckout) {
        Write-Warning "ggml: could not list the releases at $GitUrl (offline?); using existing checkout"
        exit 0
    }
    else {
        throw "ggml: could not list the releases at $GitUrl (offline?); set TENSORSHARP_GGML_GIT_REF to a tag, branch or commit"
    }
}

if ($HasCheckout) {

    Write-Host "ggml: updating existing checkout to $GitRef ($GitUrl)"
    # Do not redirect git's stderr (no `2>$null`): git writes normal progress and
    # summaries ("From github.com/...") to stderr, and under
    # $ErrorActionPreference='Stop' redirecting a native command's stderr turns that
    # output into a terminating NativeCommandError even on success (Windows
    # PowerShell 5.1). Letting it pass through keeps the $LASTEXITCODE checks below
    # working for both the success and offline (non-zero exit) paths.
    git -C $GgmlDir remote set-url origin $GitUrl
    git -C $GgmlDir fetch --depth 1 origin $GitRef
    if ($LASTEXITCODE -eq 0) {
        git -C $GgmlDir reset --hard FETCH_HEAD
        if ($LASTEXITCODE -ne 0) { throw "git reset failed" }
        $sha = (git -C $GgmlDir rev-parse --short HEAD).Trim()
        Write-Host "ggml: now at $sha"
    }
    else {
        Write-Warning "ggml: could not fetch $GitRef (offline?); using existing checkout"
    }
    exit 0
}

# No checkout yet: clear a partial/empty directory left by a failed clone.
if (Test-Path $GgmlDir) {
    Remove-Item -Recurse -Force $GgmlDir
}

Write-Host "ggml: cloning $GitUrl ($GitRef) into $GgmlDir"
# No `2>$null` here either: git clone reports progress on stderr, which would
# otherwise become a terminating error under $ErrorActionPreference='Stop'.
git clone --depth 1 --branch $GitRef $GitUrl $GgmlDir
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
}
finally {
    if ($HasFetchMutex) {
        $FetchMutex.ReleaseMutex()
    }
    $FetchMutex.Dispose()
}
