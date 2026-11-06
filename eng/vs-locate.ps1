# Locate the Visual Studio installation that carries the MSVC C++ toolset for the
# native build's target architecture: x64, or ARM64 on Windows on Arm.
#
# Why this exists instead of a bare `vswhere -latest`: the VS installer flags an
# instance "incomplete" after an interrupted update or an unfinished repair, and
# `vswhere -latest` silently skips such instances - it prints nothing and still
# exits 0 - even though cl.exe, vcvars64.bat and the CUDA MSBuild integration on
# disk all work fine. Callers that trusted `-latest` therefore concluded "no
# Visual Studio here" and degraded badly rather than visibly: CMake could not use
# a "Visual Studio NN" generator and fell back to the *serial* NMake Makefiles
# generator, which ignores `--parallel` and compiles ggml's ~190 CUDA
# translation units one at a time.
#
# Resolution order:
#   1. TENSORSHARP_VS_INSTALL_DIR - explicit override, wins outright
#   2. vswhere -all -prerelease   - relaxed query that does see incomplete and
#                                   preview instances (newest version first)
#   3. filesystem probe           - the standard install roots, newest year
#                                   first, for the target's vcvars script
#
# Dot-source this file to get Get-VisualStudioInstallation, or run it directly
# to print what was found. Returns $null when no MSVC toolset can be located.

$ErrorActionPreference = "Stop"

# The operating system's own architecture: "arm64" on Windows on Arm, else "x64".
#
# Read from the registry rather than from this process. An x64 PowerShell (or an
# x64 dotnet that launched it) runs under emulation on Windows on Arm and sees
# PROCESSOR_ARCHITECTURE=AMD64, while Windows PowerShell 5.1 - what MSBuild's Exec
# starts - has no RuntimeInformation.OSArchitecture to ask instead.
function Get-OSArchitecture {
    $machine = ""
    try {
        $machine = (Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" `
            -Name PROCESSOR_ARCHITECTURE -ErrorAction Stop).PROCESSOR_ARCHITECTURE
    }
    catch {
        $machine = $env:PROCESSOR_ARCHITECTURE
    }

    if ("$machine".Trim() -eq "ARM64") {
        return "arm64"
    }
    return "x64"
}

# The architecture the native library is built for: "x64" or "arm64".
#
# TENSORSHARP_GGML_NATIVE_ARCH chooses it explicitly. Otherwise it is the
# operating system's, which is what a plain `dotnet build` runs on: GgmlOps.dll is
# loaded into that process, so it has to match it. An x64 library on an ARM64
# machine (TENSORSHARP_GGML_NATIVE_ARCH=x64) is for an x64 .NET under emulation.
function Get-NativeBuildArchitecture {
    $requested = "$env:TENSORSHARP_GGML_NATIVE_ARCH".Trim().ToLowerInvariant()
    if ([string]::IsNullOrWhiteSpace($requested)) {
        return Get-OSArchitecture
    }
    if ($requested -in @("x64", "amd64")) {
        return "x64"
    }
    if ($requested -eq "arm64") {
        return "arm64"
    }

    throw "TENSORSHARP_GGML_NATIVE_ARCH='$env:TENSORSHARP_GGML_NATIVE_ARCH' is not supported; use x64 or arm64."
}

# The MSVC pieces that differ per target architecture.
#
# An x64 target always uses vcvars64.bat, whose x64-hosted compilers also run
# under emulation on Windows on Arm. An ARM64 target uses the ARM64-hosted
# compilers and needs an ARM64 machine: ggml probes the CPU's features at
# configure time (GGML_NATIVE), which a cross build cannot do.
function Get-MsvcTargetToolset([string] $Architecture) {
    switch ($Architecture) {
        "x64" {
            return [pscustomobject]@{
                Architecture  = "x64"
                Component     = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
                VcVarsName    = "vcvars64.bat"
                BinSubdir     = "bin\Hostx64\x64"
                PromptName    = "x64 Native Tools"
                CMakePlatform = "x64"
            }
        }
        "arm64" {
            return [pscustomobject]@{
                Architecture  = "arm64"
                Component     = "Microsoft.VisualStudio.Component.VC.Tools.ARM64"
                VcVarsName    = "vcvarsarm64.bat"
                BinSubdir     = "bin\Hostarm64\arm64"
                PromptName    = "ARM64 Native Tools"
                CMakePlatform = "ARM64"
            }
        }
        default {
            throw "Unsupported MSVC target architecture '$Architecture'; use x64 or arm64."
        }
    }
}

function Get-VisualStudioGeneratorName([string] $Version) {
    # Map the installed VS major version to its CMake generator name; a
    # hardcoded generator breaks on machines that only carry a different VS
    # (e.g. GitHub's windows-latest image now ships Visual Studio 2026 only).
    switch (("$Version".Trim() -split '\.')[0]) {
        "16" { return "Visual Studio 16 2019" }
        "17" { return "Visual Studio 17 2022" }
        "18" { return "Visual Studio 18 2026" }
        # Unknown/newer VS: return nothing and let CMake pick its own default.
        default { return "" }
    }
}

function Get-VisualStudioMajorFromDirectoryName([string] $Name) {
    # The installer's directory under "Microsoft Visual Studio" is the release
    # *year* through VS 2022 ("2019", "2022") but the *major version* from
    # VS 2026 on ("18" - the layout discussion #130's reporter pasted:
    # "C:\Program Files\Microsoft Visual Studio\18\Community"). Normalise both
    # shapes to the major version so callers can compare and sort them uniformly.
    # Returns 0 when the name is neither.
    switch ("$Name".Trim()) {
        "2017" { return 15 }
        "2019" { return 16 }
        "2022" { return 17 }
        "2026" { return 18 }
    }

    if ("$Name".Trim() -match '^\d{1,3}$') {
        return [int] $Name
    }

    return 0
}

function Get-VisualStudioVersionFromPath([string] $Path) {
    # The filesystem probe has no version metadata, only the installer's
    # directory name ("...\Microsoft Visual Studio\2022\Community",
    # "...\Microsoft Visual Studio\18\Community").
    if ("$Path" -match '\\Microsoft Visual Studio\\([^\\]+)\\') {
        $major = Get-VisualStudioMajorFromDirectoryName $Matches[1]
        if ($major -gt 0) {
            return "$major.0"
        }
    }

    return ""
}

function New-VisualStudioInstallation([string] $Path, [string] $Version, [string] $Architecture = "x64") {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }

    # The target's vcvars script is the thing callers actually need, and its
    # compiler directory is the proof that the toolset is really on disk, whatever
    # the installer's state flag says. Both are checked: the vcvars scripts for
    # every architecture can be present while only one toolset is installed.
    $toolset = Get-MsvcTargetToolset $Architecture
    $vcvars = Join-Path $Path "VC\Auxiliary\Build\$($toolset.VcVarsName)"
    if (-not (Test-Path $vcvars)) {
        return $null
    }
    $compilers = @(Get-ChildItem -Path (Join-Path $Path "VC\Tools\MSVC") -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "$($toolset.BinSubdir)\cl.exe") })
    if ($compilers.Count -eq 0) {
        return $null
    }

    if ([string]::IsNullOrWhiteSpace($Version)) {
        $Version = Get-VisualStudioVersionFromPath $Path
    }

    return [pscustomobject]@{
        Path         = $Path
        Version      = $Version
        Generator    = Get-VisualStudioGeneratorName $Version
        Architecture = $toolset.Architecture
        VcVars       = $vcvars
    }
}

function Find-VisualStudioViaVswhere([string] $Architecture = "x64") {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        return $null
    }

    # -all -prerelease lifts the default filters that hide incomplete and
    # preview instances. -format json keeps installationPath paired with
    # installationVersion on machines carrying several instances, which separate
    # -property queries would not.
    $toolset = Get-MsvcTargetToolset $Architecture
    $output = & $vswhere -all -prerelease -products * -requires $toolset.Component -format json
    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    $text = ($output | Out-String).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $null
    }

    try {
        $instances = @($text | ConvertFrom-Json)
    }
    catch {
        return $null
    }

    $sorted = $instances | Sort-Object -Property @{ Expression = {
        try { [version] $_.installationVersion } catch { [version] "0.0" }
    } } -Descending

    foreach ($instance in $sorted) {
        $found = New-VisualStudioInstallation $instance.installationPath $instance.installationVersion $Architecture
        if ($null -ne $found) {
            return $found
        }
    }

    return $null
}

function Find-VisualStudioViaFilesystem([string] $Architecture = "x64") {
    $roots = @($env:ProgramFiles, ${env:ProgramFiles(x86)}) |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { Join-Path $_ "Microsoft Visual Studio" }

    foreach ($root in $roots) {
        if (-not (Test-Path $root)) {
            continue
        }

        # Both directory shapes ("2022" and "18") normalise to a major version,
        # so newest-first sorting keeps working across the VS 2026 rename - a
        # plain string sort would rank "2022" above "18".
        $versionDirs = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
            Where-Object { (Get-VisualStudioMajorFromDirectoryName $_.Name) -gt 0 } |
            Sort-Object -Property @{ Expression = { Get-VisualStudioMajorFromDirectoryName $_.Name } } -Descending
        foreach ($yearDir in $versionDirs) {
            # Any edition will do (Community/Professional/Enterprise/BuildTools);
            # they all ship the same vcvars scripts and MSVC toolsets.
            $editionDirs = Get-ChildItem -Path $yearDir.FullName -Directory -ErrorAction SilentlyContinue
            foreach ($editionDir in $editionDirs) {
                $found = New-VisualStudioInstallation $editionDir.FullName "" $Architecture
                if ($null -ne $found) {
                    return $found
                }
            }
        }
    }

    return $null
}

# The Visual Studio installation carrying the MSVC toolset that targets
# $Architecture ("x64" or "arm64"), or $null when there is none.
function Get-VisualStudioInstallation([string] $Architecture = "x64") {
    if (-not [string]::IsNullOrWhiteSpace($env:TENSORSHARP_VS_INSTALL_DIR)) {
        $override = New-VisualStudioInstallation $env:TENSORSHARP_VS_INSTALL_DIR "" $Architecture
        if ($null -ne $override) {
            return $override
        }

        $toolset = Get-MsvcTargetToolset $Architecture
        Write-Warning ("TENSORSHARP_VS_INSTALL_DIR='$env:TENSORSHARP_VS_INSTALL_DIR' has no MSVC $Architecture toolset " +
            "(VC\Auxiliary\Build\$($toolset.VcVarsName) and $($toolset.BinSubdir)\cl.exe); ignoring it and probing " +
            "for Visual Studio normally.")
    }

    $found = Find-VisualStudioViaVswhere $Architecture
    if ($null -ne $found) {
        return $found
    }

    return Find-VisualStudioViaFilesystem $Architecture
}

# Resolve the cmake to drive the native build.
#
# CMake is a hard prerequisite that nothing here installs, and the failure mode
# when it is missing is a bare "cmake : The term 'cmake' is not recognized ..."
# from whichever script reached it first - issue #166, where the reporter had to
# work out for themselves that CMake was what they were missing. Prefer a cmake
# on PATH; otherwise fall back to the one Visual Studio's "C++ CMake tools for
# Windows" component ships (the same component that provides the ninja found by
# Find-NinjaProgram), which is how a VS-only machine has a working CMake without
# ever installing one. Returns "" when neither exists, so callers can fail with
# an actionable message.
function Find-CMakeProgram([object] $VisualStudio) {
    $onPath = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($null -ne $onPath) {
        return $onPath.Source
    }

    if ($null -ne $VisualStudio) {
        $bundled = Join-Path $VisualStudio.Path "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        if (Test-Path $bundled) {
            return $bundled
        }
    }

    return ""
}

# Find-CMakeProgram, or throw with an actionable message when there is no cmake.
function Get-RequiredCMakeProgram([object] $VisualStudio) {
    $cmakeProgram = Find-CMakeProgram $VisualStudio
    if ([string]::IsNullOrWhiteSpace($cmakeProgram)) {
        throw ("CMake was not found. The native GGML library is configured and built with CMake 3.20+, " +
            "which is a prerequisite this build does not install for you. Install it from " +
            "https://cmake.org/download/ (tick 'Add CMake to the system PATH'), or add the " +
            "'C++ CMake tools for Windows' component to Visual Studio - it ships both cmake.exe and " +
            "ninja.exe and this script will find that copy without any PATH changes.")
    }

    return $cmakeProgram
}

# Report the target architecture ("x86" / "x64" / "arm64") of the MSVC developer
# environment already active in this process, or "" when none is active or it
# cannot be determined.
#
# Only an environment for the build's target (x64, or arm64 on Windows on Arm) is
# usable here, and "is VCToolsInstallDir set?" is not enough to tell: the
# "Developer PowerShell for VS" and "Developer Command Prompt" shortcuts default
# to the **x86** toolset and set VCToolsInstallDir
# exactly like the "x64 Native Tools" prompt does. Inheriting that x86
# environment builds the whole native library 32-bit, which fails deep inside
# ggml's Vulkan backend rather than at configure time: a 32-bit target leaves
# VK_USE_64_BIT_PTR_DEFINES at 0, so vulkan.hpp makes every handle's conversion
# operator `explicit` (VULKAN_HPP_TYPESAFE_EXPLICIT) and ordinary uses of
# vk::Buffer stop compiling - "no operator found which takes a left-hand operand
# of type 'std::basic_ostream'", "'vk::CommandBuffer::copyBuffer': no matching
# overloaded function". See https://github.com/zhongkaifu/TensorSharp/discussions/130
function Get-ActiveVcTargetArchitecture {
    # VsDevCmd.bat records the target it set up here, so this is authoritative
    # for every environment that came from a VS prompt or from vcvars - including
    # one Import-VcVarsEnvironment just applied. It is also read straight from
    # the process environment, which keeps it correct immediately after an
    # import, whereas a PATH-based probe depends on PowerShell's command cache
    # having noticed the new PATH.
    #
    # `Platform` is deliberately not consulted: Import-VcVarsEnvironment drops it
    # (see below), so it is absent even after a perfectly good x64 import.
    if (-not [string]::IsNullOrWhiteSpace($env:VSCMD_ARG_TGT_ARCH)) {
        return "$env:VSCMD_ARG_TGT_ARCH".Trim().ToLowerInvariant()
    }

    # Fallbacks for an MSVC environment assembled by something other than
    # VsDevCmd (a CI action that exports only part of the environment, a
    # hand-rolled setup). The toolset lays its compilers out under
    # bin\Host<host>\<target>, so the first such directory on PATH is the one a
    # compiler-driven generator (Ninja, NMake) would pick cl.exe from. Matching
    # PATH directly rather than resolving cl.exe keeps this correct even when
    # PowerShell's command cache has not yet noticed a PATH change.
    foreach ($entry in ("$env:PATH" -split ';')) {
        if ("$entry" -match '\\bin\\Host[^\\]+\\([^\\]+)\\?$') {
            return $Matches[1].ToLowerInvariant()
        }
    }

    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cl -and "$($cl.Source)" -match '\\Host[^\\]+\\([^\\]+)\\cl\.exe$') {
        return $Matches[1].ToLowerInvariant()
    }

    return ""
}

# Import the MSVC command-line environment (PATH/INCLUDE/LIB/LIBPATH) into the
# current process, the way an "x64 Native Tools" (or "ARM64 Native Tools") prompt
# does. Generators that invoke the compilers directly (Ninja, NMake) need it; the
# Visual Studio generator does not, because MSBuild sets it up per project.
# $Architecture is the target $VcVarsPath sets up: "x64" or "arm64".
#
# The variables land in this PowerShell process only, so an MSBuild that shelled
# out to this script never sees them.
function Import-VcVarsEnvironment([string] $VcVarsPath, [string] $Architecture = "x64") {
    $toolset = Get-MsvcTargetToolset $Architecture
    $activeArch = Get-ActiveVcTargetArchitecture
    if (-not [string]::IsNullOrWhiteSpace($env:VCToolsInstallDir)) {
        if ($activeArch -eq $Architecture) {
            Write-Host "MSVC $Architecture environment already active (VCToolsInstallDir=$env:VCToolsInstallDir); not importing vcvars."
            return
        }

        if ([string]::IsNullOrWhiteSpace($activeArch)) {
            # A toolset is active but nothing identifies its target. Leave it
            # alone: importing would silently retarget a deliberately configured
            # environment - CI pins an older toolset with
            # `ilammy/msvc-dev-cmd@v1 toolset: "14.44"` because older CUDA toolkits
            # reject VS 2026's default 14.5x host compiler, and re-running vcvars64
            # would drag it back to the default. Warn instead, so a genuinely
            # 32-bit environment still leaves a trace to explain a later failure.
            Write-Warning ("An MSVC environment is active (VCToolsInstallDir=$env:VCToolsInstallDir) but its target " +
                "architecture could not be determined; assuming $Architecture and using it as-is. This native build " +
                "targets $Architecture - if the compile fails inside ggml's Vulkan backend, re-run from a plain " +
                "PowerShell so this script can set up the $Architecture environment itself.")
            return
        }
    }

    if ([string]::IsNullOrWhiteSpace($VcVarsPath) -or -not (Test-Path $VcVarsPath)) {
        throw "$($toolset.VcVarsName) not found at '$VcVarsPath'."
    }

    if (-not [string]::IsNullOrWhiteSpace($env:VCToolsInstallDir)) {
        # Active and targeting something else - import the target's environment
        # over it instead of silently producing a 32-bit (or wrong-architecture)
        # build.
        Write-Host ("MSVC environment already active but targets '$activeArch', not $Architecture " +
            "(the 'Developer PowerShell for VS' and 'Developer Command Prompt' shortcuts default to x86); " +
            "importing the $Architecture environment over it - this native build targets $Architecture.")
    }

    Write-Host "Importing MSVC environment from $VcVarsPath"
    # vcvars' sub-scripts shell out to a bare `vswhere.exe`, which only resolves
    # when the VS Installer directory is on PATH (it is inside a VS developer
    # prompt, but not in a plain shell). Without it vcvars prints "'vswhere.exe'
    # is not recognized ..." and skips part of its detection, so put it on PATH
    # for the duration of the import.
    $installerDir = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
    $originalPath = $env:PATH
    if ((Test-Path $installerDir) -and $env:PATH -notlike "*$installerDir*") {
        $env:PATH = "$installerDir;$env:PATH"
    }

    # vcvars is not re-entrant: VsDevCmd.bat keys off VSCMD_VER and declines to
    # initialise a second time in a shell that already carries a developer
    # environment. The child cmd.exe below inherits this process's variables, so
    # when we are importing x64 *over* an active x86 environment that guard would
    # make the import a silent no-op and leave the 32-bit toolset in place.
    # Clear that state for the child only, and with it the stale x86
    # INCLUDE/LIB/LIBPATH search paths that vcvars would otherwise append to.
    #
    # `set "VAR="` (quoted) is the form that really unsets: `set VAR=` before an
    # `&` separator captures the space and leaves VAR set to " ", which still
    # trips `if defined`. PATH is deliberately left alone - vcvars *prepends* its
    # x64 tool directories, so they win over any x86 ones already there, and
    # rewriting it here would drop the Installer directory prepended above.
    $childState = @(
        "VSCMD_VER", "VSCMD_ARG_HOST_ARCH", "VSCMD_ARG_TGT_ARCH", "VSCMD_ARG_APP_PLAT",
        "__VSCMD_PREINIT_PATH", "VCToolsInstallDir", "VCToolsVersion", "VCINSTALLDIR",
        "INCLUDE", "LIB", "LIBPATH"
    ) | ForEach-Object { "set `"$_=`"" }

    try {
        # cmd /s /c "<resets> & <quoted bat> && set": /s makes cmd strip only the
        # outermost quote pair, which keeps the quoted path (spaces) intact. `&`
        # binds looser than `&&`, so the resets run unconditionally and `set` still
        # runs only if vcvars succeeded - keeping the exit code checked below.
        $lines = & cmd.exe /s /c "$($childState -join ' & ') & `"$VcVarsPath`" && set"
    }
    finally {
        # vcvars' own PATH (captured in $lines) is applied below; drop the
        # temporary prepend so it cannot end up duplicated.
        $env:PATH = $originalPath
    }

    if ($LASTEXITCODE -ne 0) {
        throw "$($toolset.VcVarsName) failed with exit code $LASTEXITCODE"
    }

    $applied = 0
    foreach ($line in $lines) {
        if ("$line" -notmatch '^([^=]+)=(.*)$') {
            continue
        }

        $name = $Matches[1]
        $value = $Matches[2]
        # Platform=x64 is a vcvars artifact that nothing in a compiler-driven
        # build reads, but any nested MSBuild would - it overrides the project's
        # own platform and breaks managed builds ("Any CPU"). Drop it.
        if ($name -eq "Platform") {
            continue
        }

        Set-Item -Path "Env:$name" -Value $value
        $applied++
    }

    if ($applied -eq 0) {
        throw "$($toolset.VcVarsName) produced no environment variables."
    }

    # Confirm the import actually landed the target's toolset. Without this a
    # surprise here (a mangled PATH, a vcvars that silently no-opped) degrades into
    # a 32-bit build whose only symptom is a wall of vulkan.hpp template errors
    # hundreds of lines into the compile.
    $importedArch = Get-ActiveVcTargetArchitecture
    if (-not [string]::IsNullOrWhiteSpace($importedArch) -and $importedArch -ne $Architecture) {
        throw ("Imported '$VcVarsPath' but the MSVC environment still targets '$importedArch' rather than " +
            "$Architecture. Run this build from a plain PowerShell (not a Developer Prompt) so the script can set " +
            "up the environment itself, or from an '$($toolset.PromptName)' prompt.")
    }
}

# Direct invocation (not dot-sourced): report what was found.
if ($MyInvocation.InvocationName -ne '.') {
    $architecture = Get-NativeBuildArchitecture
    $installation = Get-VisualStudioInstallation $architecture
    if ($null -eq $installation) {
        Write-Warning "No Visual Studio installation with the MSVC $architecture C++ toolset was found."
        exit 1
    }

    $installation | Format-List | Out-String | Write-Host
}
