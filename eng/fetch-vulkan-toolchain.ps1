# Provision the Vulkan build toolchain needed by the ggml-vulkan backend
# (TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON) into ExternalProjects/vulkan-toolchain.
#
# ggml-vulkan needs three things at build time that a plain Windows box does not
# have (the Vulkan *runtime* ships with the GPU driver, the *SDK* does not):
#   1. Vulkan headers (vulkan_core.h / vulkan.hpp)         -> KhronosGroup/Vulkan-Headers
#   2. A vulkan-1 import library to link against            -> generated from the
#      system C:\Windows\System32\vulkan-1.dll with dumpbin/lib (MSVC)
#   3. The glslc GLSL->SPIR-V compiler                      -> pinned Google shaderc CI prebuilt
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
$ShadercDir = Join-Path $ToolchainDir "shaderc"
$ShadercArtifactId = "windows-vs2022-amd64-release/continuous/36/20260731-122731"
$ShadercArchiveUrl = "https://storage.googleapis.com/shaderc/artifacts/prod/graphics_shader_compiler/shaderc/$ShadercArtifactId/install.zip?generation=1785527613445501"
$ShadercArchiveSha256 = "E311E9B6872C099089FCD00C7A596BA04B9F8F8162F215421DE6C24B4D90AFE9"
$ShadercExecutableSha256 = "D35943F6DED799E16CA0280205EF151A20ABCFC9177ECC3FB924AA1B66915189"
$ShadercArtifactStamp = Join-Path $ShadercDir "tensorsharp-artifact.txt"
$ShadercBackupDir = Join-Path $ToolchainDir "shaderc-replacement-backup"
$VulkanSdkIncompatibleMarker = Join-Path $ToolchainDir "vulkan-sdk-glslc-incompatible.marker"

# Get-VisualStudioInstallation / Import-VcVarsEnvironment. A bare
# `vswhere -latest` cannot be trusted (it silently skips installer instances
# flagged incomplete); see eng/vs-locate.ps1.
. (Join-Path $ScriptDir "vs-locate.ps1")

function Test-Truthy([string] $Value) {
    return $Value -match '^(1|ON|on|On|TRUE|true|True|YES|yes|Yes)$'
}

function Invoke-CMakeDefanged([string] $FailureMessage, [string[]] $Arguments) {
    # Resolved per call (rather than once at script scope) because the only
    # caller sits behind a Test-Path guard - a toolchain that is already
    # provisioned must not need a cmake at all. Throws with install instructions
    # when there is none; see issue #166.
    $cmakeProgram = Get-RequiredCMakeProgram $VisualStudio

    # This script usually runs inside an MSBuild Exec task, whose logger scans
    # every output line for the canonical MSBuild error/warning format and fails
    # the managed build when one matches - even when the caller catches the
    # failure and deliberately degrades to a CPU/CUDA-only build. CMake's
    # "CMake Error: ..." diagnostics match that format, so capture the output
    # and re-emit it with the "<keyword> :" pattern broken.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $cmakeProgram @Arguments 2>&1
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    foreach ($line in @($output)) {
        Write-Host ("$line" -replace '(?i)\b(error|warning)(\s+[A-Z0-9]+)?\s*:', '$1$2 -')
    }

    if ($LASTEXITCODE -ne 0) { throw $FailureMessage }
}

function Test-VulkanSdkFilesComplete {
    $sdk = $env:VULKAN_SDK
    if ([string]::IsNullOrWhiteSpace($sdk)) { return $false }
    return (Test-Path (Join-Path $sdk "Include\vulkan\vulkan.h")) -and
           (Test-Path (Join-Path $sdk "Lib\vulkan-1.lib")) -and
           (Test-Path (Join-Path $sdk "Bin\glslc.exe"))
}

function Test-GlslcCompatible([string] $CompilerPath) {
    if (-not (Test-Path $CompilerPath)) { return $false }

    # ggml specializes many Vulkan workgroup sizes through local_size_*_id and
    # optimizes the resulting SPIR-V. shaderc continuous build 38 regressed this
    # exact combination: compilation succeeds, but its integrated optimizer
    # rejects LocalSizeId for a Vulkan 1.2 target. Keep the probe semantic so a
    # corrupt or otherwise incompatible compiler is rejected before CMake starts
    # generating hundreds of shaders.
    $probeSource = @(
        "#version 450",
        "layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;",
        "void main() {}"
    )
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $probeOutput = $probeSource | & $CompilerPath -fshader-stage=compute --target-env=vulkan1.2 -O - -o NUL 2>&1
        $probeExitCode = $LASTEXITCODE
    }
    catch {
        $probeExitCode = 1
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    return $probeExitCode -eq 0
}

function Test-VulkanSdkComplete {
    if (-not (Test-VulkanSdkFilesComplete)) { return $false }
    return Test-GlslcCompatible (Join-Path $env:VULKAN_SDK "Bin\glslc.exe")
}

function Test-PinnedShaderc {
    $compilerPath = Join-Path $ShadercDir "bin\glslc.exe"
    if (-not (Test-Path $compilerPath)) { return $false }

    try {
        $compilerHash = (Get-FileHash -Path $compilerPath -Algorithm SHA256).Hash
    }
    catch {
        return $false
    }
    if ($compilerHash -ne $ShadercExecutableSha256 -or -not (Test-GlslcCompatible $compilerPath)) {
        return $false
    }

    # Backfill the marker for toolchains downloaded before the artifact was
    # pinned. The executable hash above proves that this is the same build.
    $stampedArtifact = if (Test-Path $ShadercArtifactStamp) { (Get-Content $ShadercArtifactStamp -Raw).Trim() } else { "" }
    if ($stampedArtifact -ne $ShadercArtifactId) {
        Set-Content -Path $ShadercArtifactStamp -Encoding ascii -Value $ShadercArtifactId
    }
    return $true
}

function Test-ToolchainComplete {
    return (Test-Path (Join-Path $ToolchainDir "Vulkan-Headers\include\vulkan\vulkan.h")) -and
           (Test-Path (Join-Path $ToolchainDir "loader\vulkan-1.lib")) -and
           (Test-PinnedShaderc) -and
           (Test-Path (Join-Path $ToolchainDir "spirv-headers-install\share\cmake\SPIRV-Headers\SPIRV-HeadersConfig.cmake"))
}

function Test-CompatibleToolchainComplete {
    return (Test-Path (Join-Path $ToolchainDir "Vulkan-Headers\include\vulkan\vulkan.h")) -and
           (Test-Path (Join-Path $ToolchainDir "loader\vulkan-1.lib")) -and
           (Test-GlslcCompatible (Join-Path $ShadercDir "bin\glslc.exe")) -and
           (Test-Path (Join-Path $ToolchainDir "spirv-headers-install\share\cmake\SPIRV-Headers\SPIRV-HeadersConfig.cmake"))
}

# A same-volume directory rename keeps the previous compiler recoverable while
# the validated replacement is installed. If a prior process was interrupted
# between those two renames, finish the rollback (or cleanup) before deciding
# whether the toolchain is complete.
if (Test-Path $ShadercBackupDir) {
    if (-not (Test-Path $ShadercDir)) {
        Write-Host "vulkan-toolchain: restoring shaderc cache after an interrupted replacement"
        Move-Item $ShadercBackupDir $ShadercDir
    }
    elseif (Test-PinnedShaderc) {
        Remove-Item -Recurse -Force $ShadercBackupDir
    }
    else {
        Remove-Item -Recurse -Force $ShadercDir
        Move-Item $ShadercBackupDir $ShadercDir
    }
}

if (Test-VulkanSdkComplete) {
    if (Test-Path $VulkanSdkIncompatibleMarker) { Remove-Item -Force $VulkanSdkIncompatibleMarker }
    Write-Host "vulkan-toolchain: using installed Vulkan SDK at $env:VULKAN_SDK"
    exit 0
}
if (Test-VulkanSdkFilesComplete) {
    New-Item -ItemType Directory -Force -Path $ToolchainDir | Out-Null
    Set-Content -Path $VulkanSdkIncompatibleMarker -Encoding ascii -Value $env:VULKAN_SDK
    Write-Host "vulkan-toolchain: installed Vulkan SDK glslc failed the Vulkan 1.2 compatibility probe; using the portable compiler"
}
elseif (Test-Path $VulkanSdkIncompatibleMarker) {
    Remove-Item -Force $VulkanSdkIncompatibleMarker
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
if (Test-Truthy $env:TENSORSHARP_GGML_NO_UPDATE) {
    if (Test-CompatibleToolchainComplete) {
        Write-Host "vulkan-toolchain: TENSORSHARP_GGML_NO_UPDATE set; using compatible existing toolchain at $ToolchainDir"
        exit 0
    }
    $cachedCompiler = Join-Path $ShadercDir "bin\glslc.exe"
    if ((Test-Path $cachedCompiler) -and -not (Test-GlslcCompatible $cachedCompiler)) {
        throw ("TENSORSHARP_GGML_NO_UPDATE is set, but the cached glslc is incompatible with ggml's Vulkan 1.2 " +
            "shader optimization. Clear TENSORSHARP_GGML_NO_UPDATE to let TensorSharp install the pinned compiler.")
    }
}

Write-Host "vulkan-toolchain: provisioning portable Vulkan toolchain in $ToolchainDir"
New-Item -ItemType Directory -Force -Path $ToolchainDir | Out-Null

# Steps 2 and 4 need the MSVC toolset (a C/C++ compiler for the SPIRV-Headers
# configure, dumpbin/lib.exe for the vulkan-1 import library). Locate it up
# front so a box without Visual Studio fails fast instead of after the downloads.
$VisualStudio = Get-VisualStudioInstallation
if ($null -eq $VisualStudio) {
    throw ("No Visual Studio installation with the MSVC x64 C++ toolset was found; it is required to build the " +
        "SPIRV-Headers package and the vulkan-1 import library. Set TENSORSHARP_VS_INSTALL_DIR to a VS installation root.")
}

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
    # This script runs in its own powershell process, so a plain `dotnet build`
    # reaches this point without a compiler on PATH; CMake then cannot enable a
    # language ("CMAKE_C_COMPILER not set, after EnableLanguage"), because its
    # own Visual Studio discovery trusts the installer state the way
    # `vswhere -latest` does. Import vcvars and pin the NMake generator (cl.exe
    # from PATH); SPIRV-Headers is header-only, so nothing is actually compiled.
    Import-VcVarsEnvironment $VisualStudio.VcVars64
    $SpirvBuildDir = Join-Path $SpirvSrcDir "build"
    if (Test-Path $SpirvBuildDir) { Remove-Item -Recurse -Force $SpirvBuildDir }
    Invoke-CMakeDefanged "SPIRV-Headers cmake configure failed" @(
        "-S", $SpirvSrcDir, "-B", $SpirvBuildDir, "-G", "NMake Makefiles",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_INSTALL_PREFIX=$SpirvInstallDir")
    Invoke-CMakeDefanged "SPIRV-Headers cmake install failed" @("--install", $SpirvBuildDir, "--config", "Release")
}

# --- 3. glslc (pinned Google shaderc CI prebuilt) ----------------------------
if (-not (Test-PinnedShaderc)) {
    $cachedCompiler = Join-Path $ShadercDir "bin\glslc.exe"
    if (Test-Path $cachedCompiler) {
        Write-Host "vulkan-toolchain: cached glslc is not the compatible pinned build; replacing it"
    }

    # Do not follow shaderc's rolling "latest continuous" badge here. Build 38
    # includes a glslang regression that emits LocalSizeId in a form shaderc's
    # Vulkan 1.2 optimization path rejects. Build 36 predates that regression;
    # pinning it also makes clean TensorSharp builds reproducible.
    $archivePath = Join-Path $ToolchainDir "shaderc-install.zip"
    $extractDir = Join-Path $ToolchainDir "shaderc-extract"
    if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
    try {
        if (Test-Path $archivePath) {
            $cachedArchiveHash = (Get-FileHash -Path $archivePath -Algorithm SHA256).Hash
            if ($cachedArchiveHash -eq $ShadercArchiveSha256) {
                Write-Host "vulkan-toolchain: using verified cached shaderc archive at $archivePath"
            }
            else {
                Remove-Item -Force $archivePath
            }
        }
        if (-not (Test-Path $archivePath)) {
            Write-Host "vulkan-toolchain: downloading pinned glslc from $ShadercArchiveUrl"
            Invoke-WebRequest -UseBasicParsing -Uri $ShadercArchiveUrl -OutFile $archivePath
        }

        $archiveHash = (Get-FileHash -Path $archivePath -Algorithm SHA256).Hash
        if ($archiveHash -ne $ShadercArchiveSha256) {
            throw "Pinned shaderc archive checksum mismatch (expected $ShadercArchiveSha256, got $archiveHash)."
        }

        Expand-Archive -Path $archivePath -DestinationPath $extractDir
        $stagedShadercDir = Join-Path $extractDir "install"
        $stagedCompiler = Join-Path $stagedShadercDir "bin\glslc.exe"
        if (-not (Test-Path $stagedCompiler)) {
            throw "Pinned shaderc archive does not contain install\bin\glslc.exe."
        }
        $stagedCompilerHash = (Get-FileHash -Path $stagedCompiler -Algorithm SHA256).Hash
        if ($stagedCompilerHash -ne $ShadercExecutableSha256) {
            throw "Pinned glslc executable checksum mismatch (expected $ShadercExecutableSha256, got $stagedCompilerHash)."
        }
        if (-not (Test-GlslcCompatible $stagedCompiler)) {
            throw "Pinned glslc failed the Vulkan 1.2 LocalSizeId optimization compatibility probe."
        }

        # Keep the previous cache under a sibling name until the validated tree
        # is installed. The recovery block above rolls this back on the next run
        # if the process is interrupted between the two same-volume renames.
        $hadPreviousShaderc = Test-Path $ShadercDir
        if ($hadPreviousShaderc) {
            Move-Item $ShadercDir $ShadercBackupDir
        }
        try {
            Move-Item $stagedShadercDir $ShadercDir
            Set-Content -Path $ShadercArtifactStamp -Encoding ascii -Value $ShadercArtifactId
        }
        catch {
            if (Test-Path $ShadercDir) { Remove-Item -Recurse -Force $ShadercDir }
            if ($hadPreviousShaderc -and (Test-Path $ShadercBackupDir)) {
                Move-Item $ShadercBackupDir $ShadercDir
            }
            throw
        }
        if (Test-Path $ShadercBackupDir) { Remove-Item -Recurse -Force $ShadercBackupDir }
    }
    finally {
        if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
        if (Test-Path $archivePath) { Remove-Item -Force $archivePath }
    }
}

# --- 4. vulkan-1.lib import library ------------------------------------------
$LoaderDir = Join-Path $ToolchainDir "loader"
if (-not (Test-Path (Join-Path $LoaderDir "vulkan-1.lib"))) {
    $systemDll = Join-Path $env:SystemRoot "System32\vulkan-1.dll"
    if (-not (Test-Path $systemDll)) {
        throw "No Vulkan runtime found at $systemDll - install a GPU driver with Vulkan support (or the LunarG Vulkan SDK)."
    }

    $vcVersion = (Get-Content (Join-Path $VisualStudio.Path "VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt")).Trim()
    $vcBin = Join-Path $VisualStudio.Path "VC\Tools\MSVC\$vcVersion\bin\Hostx64\x64"
    if (-not (Test-Path (Join-Path $vcBin "lib.exe"))) {
        throw "MSVC binaries not found under $vcBin - the Visual Studio installation at '$($VisualStudio.Path)' has no x64 C++ toolset."
    }

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
