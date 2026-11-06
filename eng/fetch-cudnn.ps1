# Provision the cuDNN runtime used by the Wan / Qwen-Image VAE convolutions on the
# CUDA backends into ExternalProjects/cudnn.
#
# ggml lowers conv2d to im2col + mul_mat. On a full-resolution Wan VAE decode that
# lowering — not the GEMM — is the dominant cost (measured on an RTX 3080 Laptop:
# IM2COL 34%, CONCAT 17%, CONT 15%, against MUL_MAT 22%), because a 3x3 convolution
# materialises 9x its input before any math happens. cuDNN convolves directly, so
# tsg_cuda_cudnn_conv.cu replaces the whole lowering (and the band tiling that exists
# only to bound the im2col scratch) with one library call on ggml's own device
# buffers.
#
# cuDNN is NOT part of the CUDA Toolkit, so this fetches the official public
# redistributable from developer.download.nvidia.com — no account or licence click
# required for the redist channel. Only include/ and bin/ are extracted: the build
# needs the headers and the runtime dlopen()s the libraries, so the import libraries
# in lib/ are never used.
#
# Environment:
#   TENSORSHARP_CUDNN=OFF|0|false  skip entirely (the VAE stays on im2col+GEMM)
#   TENSORSHARP_CUDNN_VERSION      override the pinned cuDNN version
#   TENSORSHARP_CUDNN_CUDA_MAJOR   override the CUDA major (default: probe nvcc, else 13)
#   TS_CUDNN_DIR / CUDNN_DIR       use an existing cuDNN install and download nothing
#   TENSORSHARP_GGML_NO_UPDATE     keep whatever is already provisioned
#
# This script never throws: cuDNN is an optional accelerator, so a machine that is
# offline, behind a proxy, or short on disk must still get a working build. It
# reports what happened and leaves the tree in whatever state it managed to reach.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$CudnnDir = Join-Path $RepoRoot "ExternalProjects\cudnn"
$Marker   = Join-Path $CudnnDir ".complete"

# The pinned version. Pinned rather than "newest available" on purpose: the ggml
# fetch tracking a moving upstream is a known source of surprise rebuilds here, and
# a convolution library silently changing under a released build is worse.
$DefaultVersion = "9.25.0.15"

function Write-Step([string]$m) { Write-Host "cudnn: $m" }

function Get-Disabled {
    $v = $env:TENSORSHARP_CUDNN
    if ([string]::IsNullOrWhiteSpace($v)) { return $false }
    return @("off", "0", "false", "no") -contains $v.Trim().ToLowerInvariant()
}

function Get-CudaMajor {
    if (-not [string]::IsNullOrWhiteSpace($env:TENSORSHARP_CUDNN_CUDA_MAJOR)) {
        return $env:TENSORSHARP_CUDNN_CUDA_MAJOR.Trim()
    }
    try {
        $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
        if ($null -eq $nvcc -and (Test-Path $env:CUDA_PATH)) {
            $candidate = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
            if (Test-Path $candidate) { $nvcc = Get-Item $candidate }
        }
        if ($null -ne $nvcc) {
            $out = & $nvcc.Source --version 2>$null | Out-String
            if ($out -match "release\s+(\d+)\.") { return $Matches[1] }
        }
    } catch { }
    # TensorSharp builds only with CUDA 13.
    return "13"
}

# An existing cuDNN (system install, conda, a pip nvidia-cudnn package) wins: nothing
# to download, and the user's own version is the one they meant to test against.
function Find-ExistingCudnn {
    foreach ($d in @($env:TS_CUDNN_DIR, $env:CUDNN_DIR)) {
        if (-not [string]::IsNullOrWhiteSpace($d) -and (Test-Path (Join-Path $d "include\cudnn.h"))) {
            return (Resolve-Path $d).Path
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($env:CUDA_PATH)) {
        if (Test-Path (Join-Path $env:CUDA_PATH "include\cudnn.h")) { return (Resolve-Path $env:CUDA_PATH).Path }
    }
    return $null
}

if (Get-Disabled) {
    Write-Step "TENSORSHARP_CUDNN is off; the VAE convolutions stay on ggml im2col+GEMM."
    exit 0
}

$existing = Find-ExistingCudnn
if ($null -ne $existing) {
    Write-Step "using the cuDNN already installed at $existing"
    exit 0
}

$version = if ([string]::IsNullOrWhiteSpace($env:TENSORSHARP_CUDNN_VERSION)) { $DefaultVersion } else { $env:TENSORSHARP_CUDNN_VERSION.Trim() }
$cudaMajor = Get-CudaMajor
# The native build's target (build-windows.ps1 passes it down), else this
# process's. The ARM64 stamp names its platform so a tree that switches targets
# re-downloads instead of linking the other architecture's cuDNN; the x64 stamp
# keeps its original form, so existing x64 provisions stay valid.
$targetArch = "$env:TENSORSHARP_GGML_NATIVE_ARCH".Trim().ToLowerInvariant()
if ([string]::IsNullOrWhiteSpace($targetArch)) {
    $targetArch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x64" }
}
if ($targetArch -eq "arm64") { $plat = "windows-arm64" } else { $plat = "windows-x86_64" }
$stamp = if ($plat -eq "windows-arm64") { "$version-cuda$cudaMajor-$plat" } else { "$version-cuda$cudaMajor" }

if (Test-Path $Marker) {
    $have = (Get-Content $Marker -Raw).Trim()
    if ($have -eq $stamp) {
        Write-Step "already provisioned ($stamp)"
        exit 0
    }
    $noUpdate = @("1", "on", "true", "yes") -contains ("$env:TENSORSHARP_GGML_NO_UPDATE").Trim().ToLowerInvariant()
    if ($noUpdate) {
        Write-Step "TENSORSHARP_GGML_NO_UPDATE set; keeping the provisioned $have"
        exit 0
    }
    Write-Step "replacing provisioned $have with $stamp"
    Remove-Item -Recurse -Force $CudnnDir -ErrorAction SilentlyContinue
}

$archive = "cudnn-$plat-${version}_cuda${cudaMajor}-archive.zip"
$url = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/$plat/$archive"

try {
    New-Item -ItemType Directory -Force -Path $CudnnDir | Out-Null
    $tmp = Join-Path $CudnnDir "$archive.part"
    if (Test-Path $tmp) { Remove-Item -Force $tmp }

    Write-Step "downloading $url"
    # Stream to disk: Invoke-WebRequest buffers the whole body in memory, which is a
    # bad idea for a ~2 GB archive, and System.Net.Http is not loaded by default in
    # Windows PowerShell 5.1. HttpWebRequest exists in both editions.
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
    $req = [System.Net.HttpWebRequest]::Create($url)
    $req.Timeout = 120000
    $req.ReadWriteTimeout = 300000
    $req.UserAgent = "TensorSharp-fetch-cudnn"
    $resp = $req.GetResponse()
    try {
        $total = $resp.ContentLength
        $in = $resp.GetResponseStream()
        $out = [System.IO.File]::Create($tmp)
        try {
            $buf = New-Object byte[] (4MB)
            $done = 0L; $nextReport = 256MB
            while (($read = $in.Read($buf, 0, $buf.Length)) -gt 0) {
                $out.Write($buf, 0, $read)
                $done += $read
                if ($done -ge $nextReport) {
                    $pct = if ($total -gt 0) { " ({0:N0}%)" -f (100.0 * $done / $total) } else { "" }
                    Write-Step ("  {0:N0} MB{1}" -f ($done / 1MB), $pct)
                    $nextReport += 256MB
                }
            }
        } finally { $out.Dispose(); $in.Dispose() }
    } finally { $resp.Dispose() }
    Write-Step ("downloaded {0:N0} MB in {1:N0}s" -f ((Get-Item $tmp).Length / 1MB), $sw.Elapsed.TotalSeconds)

    # Extract include/ and bin/ only. lib/ holds the import libraries, which nothing
    # uses because the runtime resolves cuDNN with LoadLibrary/dlopen.
    Write-Step "extracting"
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($tmp)
    try {
        foreach ($e in $zip.Entries) {
            if ([string]::IsNullOrEmpty($e.Name)) { continue }   # directory entry
            # Strip the "cudnn-windows-x86_64-.../" top-level folder.
            $rel = $e.FullName -replace '^[^/]+/', ''
            if ($rel -notmatch '^(include|bin)/') { continue }
            $dest = Join-Path $CudnnDir ($rel -replace '/', '\')
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, $dest, $true)
        }
    } finally { $zip.Dispose() }
    Remove-Item -Force $tmp -ErrorAction SilentlyContinue

    if (-not (Test-Path (Join-Path $CudnnDir "include\cudnn.h"))) {
        throw "extraction finished but include\cudnn.h is missing"
    }
    Set-Content -Path $Marker -Value $stamp -Encoding ascii
    $mb = [math]::Round(((Get-ChildItem -Recurse $CudnnDir | Measure-Object Length -Sum).Sum / 1MB), 0)
    Write-Step "ready at $CudnnDir ($stamp, $mb MB)"
    Write-Step "note: the VAE uses it only with TS_VAE_CUDNN_CONV=1 - it wins on short clips"
    Write-Step "      and loses badly on long ones (see fast_conv_enabled in ggml_ops_core.cpp)."
    Write-Step "      TENSORSHARP_CUDNN=OFF skips this download entirely."
} catch {
    # Optional accelerator: report and move on. The words "error"/"warning" are kept
    # out of the message because MSBuild's Exec logger treats those as build failures
    # even when the caller tolerates a non-zero result (see fetch-vulkan-toolchain.ps1).
    Write-Step "not provisioned - $($_.Exception.Message -replace '(?i)\b(error|warning)\b', 'issue')"
    Write-Step "the VAE convolutions will use ggml im2col+GEMM; set TENSORSHARP_CUDNN=OFF to stop trying."
    Remove-Item -Force (Join-Path $CudnnDir "*.part") -ErrorAction SilentlyContinue
}
exit 0
