# Provision the Vulkan build toolchain needed by the ggml-vulkan backend
# (TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON) into ExternalProjects/vulkan-toolchain.
#
# ggml-vulkan needs three things at build time that a plain Windows box does not
# have (the Vulkan *runtime* ships with the GPU driver, the *SDK* does not):
#   1. Vulkan headers (vulkan_core.h / vulkan.hpp)         -> KhronosGroup/Vulkan-Headers
#   2. A vulkan-1 import library to link against            -> generated from the
#      system C:\Windows\System32\vulkan-1.dll with dumpbin/lib (MSVC)
#   3. The glslc GLSL->SPIR-V compiler                      -> Google shaderc CI prebuilt
# plus the SPIRV-Headers CMake package (find_package(SPIRV-Headers) in ggml).
#
# When a LunarG Vulkan SDK is installed (VULKAN_SDK env var) all of the above are
# already available and this script exits without doing anything;
# build-windows.ps1 then lets CMake's FindVulkan discover the SDK on its own.
#
# Environment overrides:
#   TENSORSHARP_GGML_NO_UPDATE  if set to 1/ON/true and the toolchain is already
#                               populated, skip all network access and reuse it
#                               (same contract as eng/fetch-ggml.ps1).
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$ToolchainDir = Join-Path $RepoRoot "ExternalProjects\vulkan-toolchain"

function Test-Truthy([string] $Value) {
    return $Value -match '^(1|ON|on|On|TRUE|true|True|YES|yes|Yes)$'
}

function Test-VulkanSdkComplete {
    $sdk = $env:VULKAN_SDK
    if ([string]::IsNullOrWhiteSpace($sdk)) { return $false }
    return (Test-Path (Join-Path $sdk "Include\vulkan\vulkan.h")) -and
           (Test-Path (Join-Path $sdk "Lib\vulkan-1.lib")) -and
           (Test-Path (Join-Path $sdk "Bin\glslc.exe"))
}

function Test-ToolchainComplete {
    return (Test-Path (Join-Path $ToolchainDir "Vulkan-Headers\include\vulkan\vulkan.h")) -and
           (Test-Path (Join-Path $ToolchainDir "loader\vulkan-1.lib")) -and
           (Test-Path (Join-Path $ToolchainDir "shaderc\bin\glslc.exe")) -and
           (Test-Path (Join-Path $ToolchainDir "spirv-headers-install\share\cmake\SPIRV-Headers\SPIRV-HeadersConfig.cmake"))
}

if (Test-VulkanSdkComplete) {
    Write-Host "vulkan-toolchain: using installed Vulkan SDK at $env:VULKAN_SDK"
    exit 0
}

if (Test-ToolchainComplete) {
    if (Test-Truthy $env:TENSORSHARP_GGML_NO_UPDATE) {
        Write-Host "vulkan-toolchain: TENSORSHARP_GGML_NO_UPDATE set; using existing toolchain at $ToolchainDir"
    }
    else {
        Write-Host "vulkan-toolchain: already populated at $ToolchainDir"
    }
    exit 0
}

Write-Host "vulkan-toolchain: provisioning portable Vulkan toolchain in $ToolchainDir"
New-Item -ItemType Directory -Force -Path $ToolchainDir | Out-Null

# --- 1. Vulkan headers -------------------------------------------------------
$HeadersDir = Join-Path $ToolchainDir "Vulkan-Headers"
if (-not (Test-Path (Join-Path $HeadersDir "include\vulkan\vulkan.h"))) {
    if (Test-Path $HeadersDir) { Remove-Item -Recurse -Force $HeadersDir }
    git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers.git $HeadersDir
    if ($LASTEXITCODE -ne 0) { throw "git clone Vulkan-Headers failed" }
}

# --- 2. SPIRV-Headers CMake package -----------------------------------------
$SpirvSrcDir = Join-Path $ToolchainDir "SPIRV-Headers"
$SpirvInstallDir = Join-Path $ToolchainDir "spirv-headers-install"
if (-not (Test-Path (Join-Path $SpirvInstallDir "share\cmake\SPIRV-Headers\SPIRV-HeadersConfig.cmake"))) {
    if (-not (Test-Path (Join-Path $SpirvSrcDir "CMakeLists.txt"))) {
        if (Test-Path $SpirvSrcDir) { Remove-Item -Recurse -Force $SpirvSrcDir }
        git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers.git $SpirvSrcDir
        if ($LASTEXITCODE -ne 0) { throw "git clone SPIRV-Headers failed" }
    }
    cmake -S $SpirvSrcDir -B (Join-Path $SpirvSrcDir "build") "-DCMAKE_INSTALL_PREFIX=$SpirvInstallDir"
    if ($LASTEXITCODE -ne 0) { throw "SPIRV-Headers cmake configure failed" }
    cmake --install (Join-Path $SpirvSrcDir "build") --config Release
    if ($LASTEXITCODE -ne 0) { throw "SPIRV-Headers cmake install failed" }
}

# --- 3. glslc (Google shaderc CI prebuilt) -----------------------------------
$ShadercDir = Join-Path $ToolchainDir "shaderc"
if (-not (Test-Path (Join-Path $ShadercDir "bin\glslc.exe"))) {
    # The badge HTML is a meta-refresh redirect to the latest continuous build's
    # install.zip; extract the target URL from it.
    $badgeUrl = "https://storage.googleapis.com/shaderc/badges/build_link_windows_vs2022_amd64_release.html"
    $badge = (Invoke-WebRequest -UseBasicParsing -Uri $badgeUrl).Content
    if ($badge -notmatch 'url=(https://[^"'']+install\.zip)') {
        throw "Could not resolve the shaderc prebuilt download URL from $badgeUrl"
    }
    $zipUrl = $Matches[1]
    $zipPath = Join-Path $ToolchainDir "shaderc-install.zip"
    Write-Host "vulkan-toolchain: downloading glslc from $zipUrl"
    Invoke-WebRequest -UseBasicParsing -Uri $zipUrl -OutFile $zipPath
    $extractDir = Join-Path $ToolchainDir "shaderc-extract"
    if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
    Expand-Archive -Path $zipPath -DestinationPath $extractDir
    if (Test-Path $ShadercDir) { Remove-Item -Recurse -Force $ShadercDir }
    # The zip contains a single top-level "install" folder with bin/lib/include.
    Move-Item (Join-Path $extractDir "install") $ShadercDir
    Remove-Item -Recurse -Force $extractDir
    Remove-Item -Force $zipPath
}

# --- 4. vulkan-1.lib import library ------------------------------------------
$LoaderDir = Join-Path $ToolchainDir "loader"
if (-not (Test-Path (Join-Path $LoaderDir "vulkan-1.lib"))) {
    $systemDll = Join-Path $env:SystemRoot "System32\vulkan-1.dll"
    if (-not (Test-Path $systemDll)) {
        throw "No Vulkan runtime found at $systemDll - install a GPU driver with Vulkan support (or the LunarG Vulkan SDK)."
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found - Visual Studio with C++ tools is required." }
    $vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($vsRoot)) { throw "No Visual Studio installation with C++ tools found." }
    $vcVersion = (Get-Content (Join-Path $vsRoot "VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt")).Trim()
    $vcBin = Join-Path $vsRoot "VC\Tools\MSVC\$vcVersion\bin\Hostx64\x64"

    New-Item -ItemType Directory -Force -Path $LoaderDir | Out-Null
    $exports = & (Join-Path $vcBin "dumpbin.exe") /exports $systemDll
    $names = $exports | Where-Object { $_ -match '^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+(\S+)' } | ForEach-Object { $Matches[1] }
    if ($names.Count -lt 10) { throw "dumpbin found only $($names.Count) exports in vulkan-1.dll; refusing to build a broken import library." }
    $defPath = Join-Path $LoaderDir "vulkan-1.def"
    Set-Content -Path $defPath -Value (@("LIBRARY vulkan-1.dll", "EXPORTS") + $names) -Encoding ascii
    & (Join-Path $vcBin "lib.exe") /nologo "/def:$defPath" /machine:x64 "/out:$(Join-Path $LoaderDir 'vulkan-1.lib')"
    if ($LASTEXITCODE -ne 0) { throw "lib.exe failed to generate vulkan-1.lib" }
}

if (-not (Test-ToolchainComplete)) {
    throw "vulkan-toolchain: provisioning finished but the toolchain is still incomplete at $ToolchainDir"
}
Write-Host "vulkan-toolchain: ready at $ToolchainDir"
