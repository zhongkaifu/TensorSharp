# Clone (or update) the upstream ggml-org/ggml repository into ExternalProjects/ggml.
#
# TensorSharp builds ggml from source: both the GGML native ops library
# (TensorSharp.GGML.Native/CMakeLists.txt) and the CUDA PTX kernels
# (TensorSharp.Backends.Cuda/native/kernels/tensorsharp_kernels.cu) consume the
# sources at ExternalProjects/ggml. The directory is not committed; it is fetched
# here at build time so the repo tracks upstream ggml.
#
# Environment overrides:
#   TENSORSHARP_GGML_GIT_URL   git URL                (default: ggml-org/ggml)
#   TENSORSHARP_GGML_GIT_REF   branch/tag/commit      (default: master, the ggml default branch)
#   TENSORSHARP_GGML_NO_UPDATE if set to 1/ON/true and a checkout already exists,
#                              skip the network fetch and use what is on disk.
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$ExternalProjectsDir = Join-Path $RepoRoot "ExternalProjects"
$GgmlDir = Join-Path $ExternalProjectsDir "ggml"

$GitUrl = if ([string]::IsNullOrWhiteSpace($env:TENSORSHARP_GGML_GIT_URL)) { "https://github.com/ggml-org/ggml.git" } else { $env:TENSORSHARP_GGML_GIT_URL }
$GitRef = if ([string]::IsNullOrWhiteSpace($env:TENSORSHARP_GGML_GIT_REF)) { "master" } else { $env:TENSORSHARP_GGML_GIT_REF }

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

if (Test-Path (Join-Path $GgmlDir ".git")) {
    if (Test-Truthy $env:TENSORSHARP_GGML_NO_UPDATE) {
        Write-Host "ggml: TENSORSHARP_GGML_NO_UPDATE set; using existing checkout at $GgmlDir"
        exit 0
    }

    Write-Host "ggml: updating existing checkout to $GitRef ($GitUrl)"
    git -C $GgmlDir remote set-url origin $GitUrl 2>$null
    git -C $GgmlDir fetch --depth 1 origin $GitRef 2>$null
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
}
finally {
    if ($HasFetchMutex) {
        $FetchMutex.ReleaseMutex()
    }
    $FetchMutex.Dispose()
}
