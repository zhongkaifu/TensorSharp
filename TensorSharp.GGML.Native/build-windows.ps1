param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $RemainingArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $ScriptDir "build-windows"

# Get-VisualStudioInstallation / Import-VcVarsEnvironment. See eng/vs-locate.ps1
# for why a bare `vswhere -latest` cannot be trusted to find the local MSVC.
. (Join-Path $ScriptDir "..\eng\vs-locate.ps1")

$EnableCuda = $env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA
$EnableVulkan = $env:TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN
$BuildTests = if ([string]::IsNullOrWhiteSpace($env:TENSORSHARP_GGML_NATIVE_BUILD_TESTS)) { "OFF" } else { $env:TENSORSHARP_GGML_NATIVE_BUILD_TESTS }
$CudaArchitectures = $env:TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES
$BuildParallelLevel = if ([string]::IsNullOrWhiteSpace($env:TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL)) { $env:CMAKE_BUILD_PARALLEL_LEVEL } else { $env:TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL }
$ExtraCMakeArgs = New-Object System.Collections.Generic.List[string]
$UserSetCudaArchitectures = $false
$UserSetGenerator = $false
$UserGenerator = ""
$UserSetPlatform = $false

function Normalize-Bool([string] $Value) {
    switch -Regex ($Value) {
        '^(ON|on|On|TRUE|true|True|YES|yes|Yes|1)$' { return "ON" }
        '^(OFF|off|Off|FALSE|false|False|NO|no|No|0)$' { return "OFF" }
        default { return "" }
    }
}

function Resolve-ExplicitCudaCompiler {
    # The nvcc the caller explicitly asked for, or "" when they did not ask.
    # CMake documents $env:CUDACXX for exactly this
    # (Modules/CMakeDetermineCUDACompiler.cmake: "prefer the environment
    # variable CUDACXX"), and -DCMAKE_CUDA_COMPILER=... is the in-band form.
    # Both were previously ignored here, so the PATH/CUDA_PATH probing below
    # silently won instead - discussion #130, where an explicit CUDA 13.3 kept
    # losing to the 12.6 on PATH.
    #
    # $ExtraCMakeArgs is filled by the argument loop further down, so this only
    # sees -DCMAKE_CUDA_COMPILER once that loop has run.
    foreach ($arg in $ExtraCMakeArgs) {
        if ("$arg" -match '^-DCMAKE_CUDA_COMPILER(:[A-Za-z]+)?=(.+)$') {
            return $Matches[2].Trim('"')
        }
    }

    $cudacxx = [Environment]::GetEnvironmentVariable("CUDACXX")
    if (-not [string]::IsNullOrWhiteSpace($cudacxx)) {
        return $cudacxx.Trim('"')
    }

    return ""
}

function Resolve-ExplicitCudaCompilerPath {
    # The explicit choice resolved to a full path, or "" when none was made.
    #
    # CMake runs the value through get_filename_component(... PROGRAM), so a bare
    # command name ("nvcc") is legal and resolves against PATH - checking only
    # Test-Path would reject it and quietly turn CUDA off. Anything that does not
    # resolve at all is an error: the caller explicitly asked for this compiler,
    # so a silent CPU-only build would be the wrong answer.
    $explicit = Resolve-ExplicitCudaCompiler
    if ([string]::IsNullOrWhiteSpace($explicit)) {
        return ""
    }

    if (Test-Path $explicit) {
        return (Resolve-Path $explicit).Path
    }

    $command = @(Get-Command $explicit -CommandType Application -ErrorAction SilentlyContinue) | Select-Object -First 1
    if ($null -ne $command) {
        return $command.Source
    }

    throw ("CUDACXX / -DCMAKE_CUDA_COMPILER names '$explicit', which is neither an existing file nor a command on " +
        "PATH. Point it at nvcc (a full path, or a bare name resolvable on PATH), or clear it - pass --no-cuda if " +
        "you want a CPU-only build.")
}

function Test-CudaToolkit {
    # An explicit choice settles it, even for a toolkit on neither PATH nor
    # CUDA_PATH - otherwise CUDA would auto-disable and the build would quietly
    # come out CPU-only. An unresolvable one throws rather than doing the same.
    if (-not [string]::IsNullOrWhiteSpace((Resolve-ExplicitCudaCompiler))) {
        Resolve-ExplicitCudaCompilerPath | Out-Null
        return $true
    }

    if (Get-Command nvcc.exe -ErrorAction SilentlyContinue) {
        return $true
    }

    foreach ($variableName in @("CUDA_PATH", "CUDA_HOME")) {
        $root = [Environment]::GetEnvironmentVariable($variableName)
        if ([string]::IsNullOrWhiteSpace($root)) {
            continue
        }

        if (Test-Path (Join-Path $root "bin\nvcc.exe")) {
            return $true
        }
    }

    return $false
}

function Resolve-CudaToolkitBinDir {
    # Returns the CUDA toolkit bin directory when nvcc.exe is not already on
    # PATH but an installed toolkit advertises itself via CUDA_PATH/CUDA_HOME
    # (the NVIDIA installer always sets CUDA_PATH). CMake's CUDA compiler
    # detection searches PATH, so without this a toolkit that is installed but
    # not on PATH fails the configure with "No CMAKE_CUDA_COMPILER could be
    # found" even though Test-CudaToolkit auto-enabled the backend.
    #
    # An explicit CUDACXX / -DCMAKE_CUDA_COMPILER wins outright: CMake resolves
    # the compiler from it, so putting a *different* toolkit's bin dir first on
    # PATH would only desync everything that probes PATH (FindCUDAToolkit,
    # eng/fetch-cudnn.ps1's CUDA-major probe, ggml's own lookups) from the
    # compiler actually in use. Return that compiler's own bin dir so they agree.
    $explicit = Resolve-ExplicitCudaCompilerPath
    if (-not [string]::IsNullOrWhiteSpace($explicit)) {
        return (Split-Path -Parent $explicit)
    }

    if (Get-Command nvcc.exe -ErrorAction SilentlyContinue) {
        return ""
    }

    foreach ($variableName in @("CUDA_PATH", "CUDA_HOME")) {
        $root = [Environment]::GetEnvironmentVariable($variableName)
        if ([string]::IsNullOrWhiteSpace($root)) {
            continue
        }

        $binDir = Join-Path $root "bin"
        if (Test-Path (Join-Path $binDir "nvcc.exe")) {
            return $binDir
        }
    }

    return ""
}

function Read-CachedBackendSetting([string] $CacheVariable) {
    $cacheFile = Join-Path $BuildDir "CMakeCache.txt"
    if (-not (Test-Path $cacheFile)) {
        return ""
    }

    $line = Select-String -Path $cacheFile -Pattern "^$([regex]::Escape($CacheVariable)):BOOL=" | Select-Object -First 1
    if ($null -eq $line) {
        return ""
    }

    return Normalize-Bool (($line.Line -split '=', 2)[1])
}

function Detect-LocalCudaArchitectures {
    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return ""
    }

    $caps = & $nvidiaSmi.Source --query-gpu=compute_cap --format=csv,noheader 2>$null
    if ($LASTEXITCODE -ne 0 -or $null -eq $caps) {
        return ""
    }

    $architectures = New-Object System.Collections.Generic.List[string]
    foreach ($cap in $caps) {
        $clean = ($cap -as [string]).Trim()
        if ($clean -notmatch '^([0-9]+)\.([0-9]+)$') {
            continue
        }

        $arch = "$($Matches[1])$($Matches[2])-real"
        if (-not $architectures.Contains($arch)) {
            $architectures.Add($arch)
        }
    }

    return ($architectures -join ';')
}

function Detect-DefaultBuildParallelLevel([bool] $CudaEnabled) {
    # Use all CPUs, bounded only by RAM. nvcc is memory-hungry, so allow ~3 GB per
    # parallel job (matches llama.cpp/ggml's own heuristic). The previous hard cap
    # of 4 jobs throttled the CUDA compile to a fraction of available cores even on
    # large machines with plenty of RAM; the memory bound below already prevents
    # OOM. Override with TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL when needed.
    #
    # Only a CUDA build needs the 3 GB bound: cl.exe peaks around a few hundred MB
    # per translation unit, so applying nvcc's budget to a CPU-only build left
    # cores idle for no reason (a 32 GB / 16-core box was capped at 10 jobs).
    $jobs = [Math]::Max(1, [Environment]::ProcessorCount)
    $bytesPerJob = if ($CudaEnabled) { 3GB } else { 1GB }

    try {
        $computer = Get-CimInstance -ClassName Win32_ComputerSystem
        if ($null -ne $computer.TotalPhysicalMemory) {
            $memoryLimitedJobs = [Math]::Max(1, [int][Math]::Floor([double]$computer.TotalPhysicalMemory / $bytesPerJob))
            $jobs = [Math]::Min($jobs, $memoryLimitedJobs)
        }
    }
    catch {
        # Keep the CPU-based default if CIM is unavailable.
    }

    return $jobs
}

function Find-NinjaProgram([object] $VisualStudio) {
    # A user-installed ninja on PATH wins.
    $onPath = Get-Command ninja.exe -ErrorAction SilentlyContinue
    if ($null -ne $onPath) {
        return $onPath.Source
    }

    # Every Visual Studio install carrying the "C++ CMake tools for Windows"
    # component ships a private ninja; CMake's own installer does not. Using it
    # means no download and no extra prerequisite for the common VS setup.
    if ($null -ne $VisualStudio) {
        $bundled = Join-Path $VisualStudio.Path "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
        if (Test-Path $bundled) {
            return $bundled
        }
    }

    return ""
}

function Read-CachedPointerSize {
    # Pointer width an existing build tree was configured for: "8" for x64, "4"
    # for a 32-bit target, "" when there is no configured tree yet.
    #
    # Note this does NOT live in CMakeCache.txt - CMAKE_SIZEOF_VOID_P is derived,
    # not cached. The persisted value is CMAKE_CXX_SIZEOF_DATA_PTR in
    # CMakeFiles/<cmake-version>/CMakeCXXCompiler.cmake, which is what
    # CMAKE_SIZEOF_VOID_P is then set from.
    $compilerFiles = @(Get-ChildItem -Path (Join-Path $BuildDir "CMakeFiles") -Recurse -Filter "CMakeCXXCompiler.cmake" -ErrorAction SilentlyContinue)
    foreach ($compilerFile in $compilerFiles) {
        $line = Select-String -Path $compilerFile.FullName -Pattern '^\s*set\(CMAKE_CXX_SIZEOF_DATA_PTR\s+"([0-9]+)"\)' | Select-Object -First 1
        if ($null -ne $line) {
            return $line.Matches[0].Groups[1].Value
        }
    }

    return ""
}

function Read-CachedCudaCompiler {
    $cacheFile = Join-Path $BuildDir "CMakeCache.txt"
    if (-not (Test-Path $cacheFile)) {
        return ""
    }

    $line = Select-String -Path $cacheFile -Pattern "^CMAKE_CUDA_COMPILER:" | Select-Object -First 1
    if ($null -eq $line) {
        return ""
    }

    return (($line.Line -split '=', 2)[1]).Trim()
}

function Read-CachedGenerator {
    $cacheFile = Join-Path $BuildDir "CMakeCache.txt"
    if (-not (Test-Path $cacheFile)) {
        return ""
    }

    $line = Select-String -Path $cacheFile -Pattern "^CMAKE_GENERATOR:INTERNAL=" | Select-Object -First 1
    if ($null -eq $line) {
        return ""
    }

    return (($line.Line -split '=', 2)[1]).Trim()
}

for ($i = 0; $i -lt $RemainingArgs.Length; $i++) {
    $arg = $RemainingArgs[$i]

    switch -Regex ($arg) {
        '^--cuda$' {
            $EnableCuda = "ON"
            continue
        }
        '^--no-cuda$' {
            $EnableCuda = "OFF"
            continue
        }
        '^--vulkan$' {
            $EnableVulkan = "ON"
            continue
        }
        '^--no-vulkan$' {
            $EnableVulkan = "OFF"
            continue
        }
        '^--tests$' {
            $BuildTests = "ON"
            continue
        }
        '^--cuda-arch=(.+)$' {
            $CudaArchitectures = $Matches[1]
            continue
        }
        '^--cuda-arch$' {
            $i++
            if ($i -ge $RemainingArgs.Length) {
                throw "Missing value for --cuda-arch"
            }
            $CudaArchitectures = $RemainingArgs[$i]
            continue
        }
        '^-DCMAKE_CUDA_ARCHITECTURES=.+$' {
            $UserSetCudaArchitectures = $true
            $ExtraCMakeArgs.Add($arg)
            continue
        }
        '^-G$' {
            $UserSetGenerator = $true
            $ExtraCMakeArgs.Add($arg)
            $i++
            if ($i -ge $RemainingArgs.Length) {
                throw "Missing value for -G"
            }
            # Remember the name too, not just that one was given: the generator
            # decides whether this script has to import the MSVC environment.
            $UserGenerator = $RemainingArgs[$i]
            $ExtraCMakeArgs.Add($RemainingArgs[$i])
            continue
        }
        '^-G.+$' {
            $UserSetGenerator = $true
            $UserGenerator = $arg.Substring(2)
            $ExtraCMakeArgs.Add($arg)
            continue
        }
        '^-A$' {
            $UserSetPlatform = $true
            $ExtraCMakeArgs.Add($arg)
            $i++
            if ($i -ge $RemainingArgs.Length) {
                throw "Missing value for -A"
            }
            $ExtraCMakeArgs.Add($RemainingArgs[$i])
            continue
        }
        '^-A.+$' {
            $UserSetPlatform = $true
            $ExtraCMakeArgs.Add($arg)
            continue
        }
        default {
            $ExtraCMakeArgs.Add($arg)
        }
    }
}

$EnableCuda = Normalize-Bool $EnableCuda
if ([string]::IsNullOrWhiteSpace($EnableCuda)) {
    $EnableCuda = Read-CachedBackendSetting "TENSORSHARP_GGML_NATIVE_ENABLE_CUDA"
}
if ([string]::IsNullOrWhiteSpace($EnableCuda) -and (Test-CudaToolkit)) {
    $EnableCuda = "ON"
}
if ([string]::IsNullOrWhiteSpace($EnableCuda)) {
    $EnableCuda = "OFF"
}

# Vulkan: an explicit choice (--vulkan/--no-vulkan or
# TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN) wins and sticks via the CMake cache;
# TENSORSHARP_GGML_NATIVE_VULKAN_EXPLICIT records that the cached value was
# explicit, so an auto-detected default never becomes sticky. Otherwise the
# backend is auto-enabled when the machine supports Vulkan (the runtime
# vulkan-1.dll that ships with every recent GPU driver is present) and the
# build toolchain (headers, loader import lib, glslc, SPIRV-Headers) is
# downloaded automatically by eng/fetch-vulkan-toolchain.ps1 when missing. An
# auto-enable whose toolchain cannot be provisioned (e.g. no network) degrades
# to a warning and builds without Vulkan; an explicit --vulkan makes it fatal.
$VulkanExplicit = "OFF"
$EnableVulkan = Normalize-Bool $EnableVulkan
if (-not [string]::IsNullOrWhiteSpace($EnableVulkan)) {
    $VulkanExplicit = "ON"
}
elseif ((Read-CachedBackendSetting "TENSORSHARP_GGML_NATIVE_VULKAN_EXPLICIT") -eq "ON") {
    $EnableVulkan = Read-CachedBackendSetting "TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN"
    if (-not [string]::IsNullOrWhiteSpace($EnableVulkan)) {
        $VulkanExplicit = "ON"
    }
}
if ([string]::IsNullOrWhiteSpace($EnableVulkan)) {
    $EnableVulkan = if (Test-Path (Join-Path $env:SystemRoot "System32\vulkan-1.dll")) { "ON" } else { "OFF" }
}

$VulkanCMakeArgs = New-Object System.Collections.Generic.List[string]
$VulkanDegraded = $false
if ($EnableVulkan -eq "ON") {
    try {
        # Ensure the Vulkan build toolchain (headers, loader import lib, glslc,
        # SPIRV-Headers) is available. With a LunarG SDK installed the script is a
        # no-op and FindVulkan discovers everything through VULKAN_SDK; otherwise it
        # provisions a portable toolchain that is passed to CMake explicitly below.
        & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ScriptDir "..\eng\fetch-vulkan-toolchain.ps1")
        if ($LASTEXITCODE -ne 0) { throw "fetch-vulkan-toolchain.ps1 failed with exit code $LASTEXITCODE" }

        $VulkanSdk = $env:VULKAN_SDK
        $VulkanSdkIncompatibleMarker = Join-Path $ScriptDir "..\ExternalProjects\vulkan-toolchain\vulkan-sdk-glslc-incompatible.marker"
        $HasFullSdk = -not [string]::IsNullOrWhiteSpace($VulkanSdk) -and
            (Test-Path (Join-Path $VulkanSdk "Include\vulkan\vulkan.h")) -and
            (Test-Path (Join-Path $VulkanSdk "Lib\vulkan-1.lib")) -and
            (Test-Path (Join-Path $VulkanSdk "Bin\glslc.exe")) -and
            -not (Test-Path $VulkanSdkIncompatibleMarker)
        if (-not $HasFullSdk) {
            $ToolchainDir = (Resolve-Path (Join-Path $ScriptDir "..\ExternalProjects\vulkan-toolchain")).Path
            $VulkanCMakeArgs.Add("-DVulkan_INCLUDE_DIR=$(Join-Path $ToolchainDir 'Vulkan-Headers\include')")
            $VulkanCMakeArgs.Add("-DVulkan_LIBRARY=$(Join-Path $ToolchainDir 'loader\vulkan-1.lib')")
            $VulkanCMakeArgs.Add("-DVulkan_GLSLC_EXECUTABLE=$(Join-Path $ToolchainDir 'shaderc\bin\glslc.exe')")
            $VulkanCMakeArgs.Add("-DSPIRV-Headers_DIR=$(Join-Path $ToolchainDir 'spirv-headers-install\share\cmake\SPIRV-Headers')")
        }
    }
    catch {
        if ($VulkanExplicit -eq "ON") { throw }
        Write-Warning ("This machine supports Vulkan but the ggml-vulkan build toolchain could not be provisioned: " +
            "$($_.Exception.Message) Building without the ggml-vulkan backend; pass --vulkan to make this an error.")
        $EnableVulkan = "OFF"
        $VulkanCMakeArgs.Clear()
        $VulkanDegraded = $true
    }
}

# Resolved only when CUDA is on: a stale CUDACXX left in the environment must not
# fail a build that is not using CUDA at all.
$ExplicitCudaCompiler = if ($EnableCuda -eq "ON") { Resolve-ExplicitCudaCompilerPath } else { "" }

if ($EnableCuda -eq "ON") {
    $CudaBinDir = Resolve-CudaToolkitBinDir
    if (-not [string]::IsNullOrWhiteSpace($CudaBinDir)) {
        if (-not [string]::IsNullOrWhiteSpace($ExplicitCudaCompiler)) {
            Write-Host "Using the CUDA toolkit at '$CudaBinDir' (from CUDACXX / -DCMAKE_CUDA_COMPILER) for this build."
        }
        else {
            Write-Host "nvcc.exe is not on PATH; using the CUDA toolkit at '$CudaBinDir' (from CUDA_PATH/CUDA_HOME) for this build."
        }

        $env:PATH = "$CudaBinDir;$env:PATH"
    }
}

if ($EnableCuda -eq "ON" -and [string]::IsNullOrWhiteSpace($CudaArchitectures) -and -not $UserSetCudaArchitectures) {
    $detectedArchitectures = Detect-LocalCudaArchitectures
    if (-not [string]::IsNullOrWhiteSpace($detectedArchitectures)) {
        $CudaArchitectures = $detectedArchitectures
    }
}

$CudaArchSummary = "n/a"
if ($EnableCuda -eq "ON") {
    if ($UserSetCudaArchitectures) {
        $CudaArchSummary = "custom via CMAKE_CUDA_ARCHITECTURES"
    }
    elseif (-not [string]::IsNullOrWhiteSpace($CudaArchitectures)) {
        $CudaArchSummary = $CudaArchitectures
    }
    else {
        $CudaArchSummary = "ggml default"
    }
}

if ([string]::IsNullOrWhiteSpace($BuildParallelLevel)) {
    $BuildParallelLevel = Detect-DefaultBuildParallelLevel ($EnableCuda -eq "ON")
}
if (($BuildParallelLevel -as [int]) -lt 1) {
    throw "Invalid build parallel level: $BuildParallelLevel"
}

# Generator choice, in order of preference:
#
#   1. Ninja - one dependency graph over every translation unit, so all ~190 CUDA
#      TUs plus the ggml/GgmlOps C++ sources saturate the requested job count
#      regardless of which CMake target they belong to. It also sidesteps the
#      MSBuild file-tracker path-length limit that CMakeLists.txt guards against,
#      and needs no CUDA MSBuild integration for the installed VS version.
#   2. The "Visual Studio NN" generator - MSBuild parallelises across projects
#      only, so ggml-cuda's TUs still compile serially, but it works without a
#      vcvars environment.
#   3. Nothing, letting CMake decide - warned about below, because CMake's own
#      fallback when it cannot see a VS instance is the strictly serial
#      NMake Makefiles generator, where `--parallel` is silently ignored.
$VisualStudio = Get-VisualStudioInstallation
$NinjaProgram = Find-NinjaProgram $VisualStudio
# Resolved up front rather than at the first `cmake` call: CMake is a hard
# prerequisite, and "not recognized as a cmdlet" 400 lines into a build does not
# tell the user what to install (issue #166).
$CMakeProgram = Get-RequiredCMakeProgram $VisualStudio

$GeneratorArgs = New-Object System.Collections.Generic.List[string]
if (-not $UserSetGenerator -and [string]::IsNullOrWhiteSpace($env:CMAKE_GENERATOR)) {
    if (-not [string]::IsNullOrWhiteSpace($NinjaProgram)) {
        $GeneratorArgs.Add("-G")
        $GeneratorArgs.Add("Ninja")
    }
    elseif ($null -ne $VisualStudio -and -not [string]::IsNullOrWhiteSpace($VisualStudio.Generator)) {
        $GeneratorArgs.Add("-G")
        $GeneratorArgs.Add($VisualStudio.Generator)
    }
}

# An explicit -G on the command line beats $env:CMAKE_GENERATOR, which beats the
# generator chosen above. A caller-supplied -G was appended to $ExtraCMakeArgs by
# the argument loop, so it is tracked separately in $UserGenerator.
$effectiveGenerator =
    if (-not [string]::IsNullOrWhiteSpace($UserGenerator)) { $UserGenerator }
    elseif (-not [string]::IsNullOrWhiteSpace($env:CMAKE_GENERATOR)) { $env:CMAKE_GENERATOR }
    else { ($GeneratorArgs | Select-Object -Last 1) }
if (-not $UserSetPlatform -and [string]::IsNullOrWhiteSpace($env:CMAKE_GENERATOR_PLATFORM) -and $effectiveGenerator -like "Visual Studio*") {
    $GeneratorArgs.Add("-A")
    $GeneratorArgs.Add("x64")
}

$GeneratorCMakeArgs = New-Object System.Collections.Generic.List[string]
if ($effectiveGenerator -like "*Ninja*") {
    # Ninja drives cl.exe/link.exe/nvcc itself, so this process needs the MSVC
    # environment. Importing it here means a plain `dotnet build` works from any
    # shell, not just from an "x64 Native Tools" prompt.
    if ($null -ne $VisualStudio) {
        Import-VcVarsEnvironment $VisualStudio.VcVars64
    }
    elseif ([string]::IsNullOrWhiteSpace($env:VCToolsInstallDir)) {
        Write-Warning ("Generator '$effectiveGenerator' needs the MSVC command-line environment but no Visual Studio " +
            "installation was found; run from an 'x64 Native Tools' prompt if the configure step cannot find a compiler.")
    }
    elseif ((Get-ActiveVcTargetArchitecture) -notin @("", "x64")) {
        # An MSVC environment is active but targets x86, and no Visual Studio was
        # found to import the x64 one over it. Say so now: the configure would
        # otherwise succeed and the build fail much later inside ggml's Vulkan
        # backend, which does not compile 32-bit (discussion #130).
        Write-Warning ("The active MSVC environment targets '$(Get-ActiveVcTargetArchitecture)', not x64, and no Visual Studio " +
            "installation was found to import the x64 environment from. This native build is x64-only - run it " +
            "from an 'x64 Native Tools' prompt, or set TENSORSHARP_VS_INSTALL_DIR to a Visual Studio installation root.")
    }

    # Pin the ninja we found: when it comes from the Visual Studio install it is
    # not on PATH, so CMake would not otherwise locate it.
    if (-not [string]::IsNullOrWhiteSpace($NinjaProgram)) {
        $GeneratorCMakeArgs.Add("-DCMAKE_MAKE_PROGRAM=$NinjaProgram")
    }
}
elseif ([string]::IsNullOrWhiteSpace($effectiveGenerator) -or $effectiveGenerator -like "*NMake*") {
    Write-Warning ("No Ninja and no usable Visual Studio generator were found, so this build may fall back to the " +
        "serial NMake Makefiles generator, which ignores --parallel and compiles one file at a time. Install the " +
        "'C++ CMake tools for Windows' VS component (it provides ninja), put ninja.exe on PATH, or set " +
        "TENSORSHARP_VS_INSTALL_DIR to a working Visual Studio installation.")
}

if ($EnableCuda -eq "ON" -and $effectiveGenerator -like "Visual Studio*" -and $null -ne $VisualStudio) {
    # The "Visual Studio NN" generators find nvcc through the CUDA toolkit's
    # MSBuild integration (CUDA *.props under the VS instance), not through
    # PATH. The CUDA installer only lays that integration down for VS versions
    # it supports - e.g. CUDA 12.x predates VS 2026 - and without it the
    # configure fails with the opaque "No CMAKE_CUDA_COMPILER could be found".
    $cudaBuildCustomizations = Join-Path $VisualStudio.Path "MSBuild\Microsoft\VC"
    $hasCudaVsIntegration = (Test-Path $cudaBuildCustomizations) -and
        @(Get-ChildItem -Path $cudaBuildCustomizations -Recurse -Filter "CUDA *.props" -ErrorAction SilentlyContinue).Count -gt 0
    if (-not $hasCudaVsIntegration) {
        Write-Warning ("CUDA is enabled and the '$effectiveGenerator' generator was selected, but the Visual Studio " +
            "instance at '$($VisualStudio.Path)' has no CUDA MSBuild integration (no 'CUDA *.props' under " +
            "MSBuild\Microsoft\VC). The configure step will likely fail with 'No CMAKE_CUDA_COMPILER could be found'. " +
            "Install the 'C++ CMake tools for Windows' VS component so this script can use the Ninja generator " +
            "(which drives nvcc directly and needs no VS integration), or install a CUDA toolkit that supports " +
            "this Visual Studio version.")
    }

    if (-not [string]::IsNullOrWhiteSpace($ExplicitCudaCompiler)) {
        # CMake reads CUDACXX / CMAKE_CUDA_COMPILER only for non-Visual-Studio
        # generators - CMakeDetermineCUDACompiler.cmake puts that whole block in
        # the `else()` of `if(CMAKE_GENERATOR MATCHES "Visual Studio")`. Without
        # this warning the configure banner would advertise a CUDA_COMPILER that
        # is silently ignored, which is the very confusion discussion #130 hit.
        #
        # -cmatch 'cuda=' catches both "-Tcuda=13.3" and the split "-T"
        # "cuda=13.3" the argument loop stores as two entries; case-sensitive so
        # an unrelated -DCMAKE_CUDA_* argument cannot match, because CMake spells
        # the toolset field lowercase.
        if (@($ExtraCMakeArgs | Where-Object { "$_" -cmatch 'cuda=' }).Count -eq 0) {
            Write-Warning ("CUDACXX / -DCMAKE_CUDA_COMPILER ('$ExplicitCudaCompiler') is ignored by the " +
                "'$effectiveGenerator' generator - CMake honours it only for non-Visual-Studio generators. " +
                "Pass '-T cuda=<version-or-path>' instead, or install the 'C++ CMake tools for Windows' VS " +
                "component so this script can use the Ninja generator, which does honour it.")
        }
    }
}

# The generator is sticky in the CMake cache and CMake refuses to reconfigure an
# existing build tree with a different one, so switching (e.g. off an older
# NMake-configured tree onto Ninja) means starting the tree over.
$CachedGenerator = Read-CachedGenerator
if (-not [string]::IsNullOrWhiteSpace($CachedGenerator) -and
    -not [string]::IsNullOrWhiteSpace($effectiveGenerator) -and
    $CachedGenerator -ne $effectiveGenerator) {
    Write-Host "Generator changed ('$CachedGenerator' -> '$effectiveGenerator'); removing $BuildDir to reconfigure from scratch."
    Remove-Item -Recurse -Force $BuildDir
}

# A tree configured 32-bit stays 32-bit: CMake pins the absolute cl.exe path and
# CMAKE_SIZEOF_VOID_P into the cache on the first configure and will not switch
# compilers afterwards. So anyone who already hit the x86 build (discussion #130)
# would apply the vcvars fix above, re-run, and get the byte-identical Vulkan
# errors. Start such a tree over so the fix actually takes effect.
$CachedPointerSize = Read-CachedPointerSize
if (-not [string]::IsNullOrWhiteSpace($CachedPointerSize) -and $CachedPointerSize -ne "8") {
    Write-Host "Existing build tree was configured for a $([int] $CachedPointerSize * 8)-bit target; removing $BuildDir to reconfigure as x64."
    Remove-Item -Recurse -Force $BuildDir
}

# CMake consults CUDACXX only when CMAKE_CUDA_COMPILER is not already cached
# (Modules/CMakeDetermineCUDACompiler.cmake: `if(NOT CMAKE_CUDA_COMPILER)`), and
# find_program() caches whatever nvcc it found on the *first* configure -
# including one that then failed. A later CUDACXX naming a different toolkit is
# therefore ignored, which is what "it was not being picked up" meant in
# discussion #130. CMake also refuses to swap a compiler inside an existing tree,
# so the only fix is to start the tree over, exactly as for a generator change.
if (-not [string]::IsNullOrWhiteSpace($ExplicitCudaCompiler)) {
    $CachedCudaCompiler = Read-CachedCudaCompiler
    # CMake writes the cache entry with forward slashes; $ExplicitCudaCompiler is
    # already a resolved full path.
    if (-not [string]::IsNullOrWhiteSpace($CachedCudaCompiler) -and
        ($CachedCudaCompiler -replace '/', '\') -ne $ExplicitCudaCompiler) {
        Write-Host "CUDA compiler changed ('$CachedCudaCompiler' -> '$ExplicitCudaCompiler'); removing $BuildDir to reconfigure from scratch."
        Remove-Item -Recurse -Force $BuildDir
    }
}

$GeneratorSummary = if ([string]::IsNullOrWhiteSpace($effectiveGenerator)) { "CMake default" } else { $effectiveGenerator }
$CudaCompilerSummary = if ([string]::IsNullOrWhiteSpace($ExplicitCudaCompiler)) { "auto" } else { $ExplicitCudaCompiler }
Write-Host "Configuring TensorSharp.GGML.Native (CUDA=$EnableCuda, CUDA_COMPILER=$CudaCompilerSummary, CUDA_ARCHITECTURES=$CudaArchSummary, VULKAN=$EnableVulkan, TESTS=$BuildTests, GENERATOR=$GeneratorSummary, PARALLEL=$BuildParallelLevel)"

$CMakeArgs = @(
    "-DCMAKE_BUILD_TYPE=Release",
    "-DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=$EnableCuda",
    "-DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=$EnableVulkan",
    "-DTENSORSHARP_GGML_NATIVE_VULKAN_EXPLICIT=$VulkanExplicit",
    "-DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=$BuildTests"
)
$CMakeArgs += $VulkanCMakeArgs

if ($EnableCuda -eq "ON" -and -not [string]::IsNullOrWhiteSpace($CudaArchitectures) -and -not $UserSetCudaArchitectures) {
    $CMakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
}

# Ensure the ggml sources are present (cloned from ggml-org/ggml at build time).
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ScriptDir "..\eng\fetch-ggml.ps1")
if ($LASTEXITCODE -ne 0) { throw "fetch-ggml.ps1 failed with exit code $LASTEXITCODE" }

# cuDNN for the VAE convolutions on the CUDA backends. Optional and best-effort:
# fetch-cudnn.ps1 never fails the build (it exits 0 even when it cannot download),
# and CMake keeps ggml's im2col+GEMM lowering when the headers are absent.
# TENSORSHARP_CUDNN=OFF skips it entirely.
if ($EnableCuda -eq "ON") {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ScriptDir "..\eng\fetch-cudnn.ps1")
}

& $CMakeProgram -S $ScriptDir -B $BuildDir @GeneratorArgs @GeneratorCMakeArgs @CMakeArgs @ExtraCMakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed with exit code $LASTEXITCODE" }

$BuildArgs = @("--build", $BuildDir, "--config", "Release", "--parallel", "$BuildParallelLevel")
if ((Normalize-Bool $BuildTests) -eq "ON") {
    & $CMakeProgram @BuildArgs
}
else {
    & $CMakeProgram @BuildArgs --target GgmlOps
}
if ($LASTEXITCODE -ne 0) { throw "cmake build failed with exit code $LASTEXITCODE" }

# Record whether an auto-enabled Vulkan backend had to be dropped. The build
# itself succeeded either way, so MSBuild would otherwise freshen its
# up-to-date stamp and skip this script on every later build - leaving a
# silently Vulkan-less GgmlOps.dll even after the cause (no network, no MSVC)
# is fixed. TensorSharp.Backends.GGML.csproj invalidates that stamp while this
# marker exists, so provisioning is retried until it succeeds.
$VulkanDegradedMarker = Join-Path $BuildDir "vulkan-degraded.marker"
if ($VulkanDegraded) {
    Set-Content -Path $VulkanDegradedMarker -Encoding ascii -Value @(
        "The ggml-vulkan backend was auto-enabled but its build toolchain could not be provisioned,",
        "so this build produced GgmlOps.dll without Vulkan support. Delete this file to stop retrying,",
        "or pass --no-vulkan to disable the backend explicitly.")
}
elseif (Test-Path $VulkanDegradedMarker) {
    Remove-Item -Force $VulkanDegradedMarker
}
