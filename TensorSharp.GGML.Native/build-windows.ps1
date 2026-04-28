param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $RemainingArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $ScriptDir "build-windows"
$EnableCuda = $env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA
$BuildTests = if ([string]::IsNullOrWhiteSpace($env:TENSORSHARP_GGML_NATIVE_BUILD_TESTS)) { "OFF" } else { $env:TENSORSHARP_GGML_NATIVE_BUILD_TESTS }
$CudaArchitectures = $env:TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES
$BuildParallelLevel = if ([string]::IsNullOrWhiteSpace($env:TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL)) { $env:CMAKE_BUILD_PARALLEL_LEVEL } else { $env:TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL }
$ExtraCMakeArgs = New-Object System.Collections.Generic.List[string]
$UserSetCudaArchitectures = $false
$UserSetGenerator = $false
$UserSetPlatform = $false

function Normalize-Bool([string] $Value) {
    switch -Regex ($Value) {
        '^(ON|on|On|TRUE|true|True|YES|yes|Yes|1)$' { return "ON" }
        '^(OFF|off|Off|FALSE|false|False|NO|no|No|0)$' { return "OFF" }
        default { return "" }
    }
}

function Test-CudaToolkit {
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

function Read-CachedCudaSetting {
    $cacheFile = Join-Path $BuildDir "CMakeCache.txt"
    if (-not (Test-Path $cacheFile)) {
        return ""
    }

    $line = Select-String -Path $cacheFile -Pattern '^TENSORSHARP_GGML_NATIVE_ENABLE_CUDA:BOOL=' | Select-Object -First 1
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

function Detect-DefaultBuildParallelLevel {
    $jobs = [Math]::Max(1, [Environment]::ProcessorCount)

    try {
        $computer = Get-CimInstance -ClassName Win32_ComputerSystem
        if ($null -ne $computer.TotalPhysicalMemory) {
            $memoryLimitedJobs = [Math]::Max(1, [int][Math]::Floor([double]$computer.TotalPhysicalMemory / (3GB)))
            $jobs = [Math]::Min($jobs, $memoryLimitedJobs)
        }
    }
    catch {
        # Keep the CPU-based default if CIM is unavailable.
    }

    if ($jobs -gt 4) {
        $jobs = 4
    }

    return $jobs
}

function Get-DefaultVisualStudioGenerator {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        return ""
    }

    $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($installPath)) {
        return ""
    }

    return "Visual Studio 17 2022"
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
            $ExtraCMakeArgs.Add($RemainingArgs[$i])
            continue
        }
        '^-G.+$' {
            $UserSetGenerator = $true
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
    $EnableCuda = Read-CachedCudaSetting
}
if ([string]::IsNullOrWhiteSpace($EnableCuda) -and (Test-CudaToolkit)) {
    $EnableCuda = "ON"
}
if ([string]::IsNullOrWhiteSpace($EnableCuda)) {
    $EnableCuda = "OFF"
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
    $BuildParallelLevel = Detect-DefaultBuildParallelLevel
}
if (($BuildParallelLevel -as [int]) -lt 1) {
    throw "Invalid build parallel level: $BuildParallelLevel"
}

$GeneratorArgs = New-Object System.Collections.Generic.List[string]
if (-not $UserSetGenerator -and [string]::IsNullOrWhiteSpace($env:CMAKE_GENERATOR)) {
    $generator = Get-DefaultVisualStudioGenerator
    if (-not [string]::IsNullOrWhiteSpace($generator)) {
        $GeneratorArgs.Add("-G")
        $GeneratorArgs.Add($generator)
    }
}

$effectiveGenerator = if (-not [string]::IsNullOrWhiteSpace($env:CMAKE_GENERATOR)) { $env:CMAKE_GENERATOR } else { ($GeneratorArgs | Select-Object -Last 1) }
if (-not $UserSetPlatform -and [string]::IsNullOrWhiteSpace($env:CMAKE_GENERATOR_PLATFORM) -and $effectiveGenerator -like "Visual Studio*") {
    $GeneratorArgs.Add("-A")
    $GeneratorArgs.Add("x64")
}

Write-Host "Configuring TensorSharp.GGML.Native (CUDA=$EnableCuda, CUDA_ARCHITECTURES=$CudaArchSummary, TESTS=$BuildTests, PARALLEL=$BuildParallelLevel)"

$CMakeArgs = @(
    "-DCMAKE_BUILD_TYPE=Release",
    "-DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=$EnableCuda",
    "-DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=$BuildTests"
)

if ($EnableCuda -eq "ON" -and -not [string]::IsNullOrWhiteSpace($CudaArchitectures) -and -not $UserSetCudaArchitectures) {
    $CMakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
}

cmake -S $ScriptDir -B $BuildDir @GeneratorArgs @CMakeArgs @ExtraCMakeArgs

$BuildArgs = @("--build", $BuildDir, "--config", "Release", "--parallel", "$BuildParallelLevel")
if ((Normalize-Bool $BuildTests) -eq "ON") {
    cmake @BuildArgs
}
else {
    cmake @BuildArgs --target GgmlOps
}
