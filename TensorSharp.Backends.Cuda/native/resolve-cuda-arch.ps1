param(
    [string]$FallbackArch = "compute_61"
)

function Normalize-Arch([string]$value) {
    if ([string]::IsNullOrWhiteSpace($value)) { return "" }
    $v = $value.Trim()
    if ($v -match '^(auto|native)$') { return "" }
    if ($v -match '^(\d+)\.(\d+)$') { return "compute_$($Matches[1])$($Matches[2])" }
    if ($v -match '^\d{2,3}$') { return "compute_$v" }
    if ($v -match '^(compute|sm)_(\d{2,3})$') { return "$($Matches[1].ToLowerInvariant())_$($Matches[2])" }
    return $v
}

function Get-ArchNumber([string]$arch) {
    if ($arch -match '^(compute|sm)_(\d{2,3})$') { return [int]$Matches[2] }
    return -1
}

$envArch = Normalize-Arch $env:TENSORSHARP_CUDA_ARCH
if (-not [string]::IsNullOrWhiteSpace($envArch)) {
    Write-Output $envArch
    exit 0
}

$fallback = Normalize-Arch $FallbackArch
if ([string]::IsNullOrWhiteSpace($fallback)) { $fallback = "compute_61" }

$supported = @()
try {
    $supported = @(nvcc --list-gpu-arch 2>$null | ForEach-Object {
        if ($_ -match 'compute_(\d{2,3})') { [int]$Matches[1] }
    } | Sort-Object -Unique)
} catch {
    $supported = @()
}

$gpuLines = @()
try {
    $gpuLines = @(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>$null)
} catch {
    $nvidiaSmi = Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    if (Test-Path $nvidiaSmi) {
        try {
            $gpuLines = @(& $nvidiaSmi --query-gpu=name,compute_cap --format=csv,noheader 2>$null)
        } catch {
            $gpuLines = @()
        }
    }
}

$gpuIndex = 0
if (-not [string]::IsNullOrWhiteSpace($env:CUDA_VISIBLE_DEVICES)) {
    $first = $env:CUDA_VISIBLE_DEVICES.Split(',')[0].Trim()
    $parsed = 0
    if ([int]::TryParse($first, [ref]$parsed) -and $parsed -ge 0 -and $parsed -lt $gpuLines.Count) {
        $gpuIndex = $parsed
    }
}

$detected = -1
if ($gpuLines.Count -gt 0 -and $gpuLines[$gpuIndex] -match ',\s*(\d+)\.(\d+)') {
    $detected = [int]("$($Matches[1])$($Matches[2])")
}

if ($detected -gt 0) {
    if ($supported.Count -eq 0 -or $supported -contains $detected) {
        Write-Output "compute_$detected"
        exit 0
    }

    $best = @($supported | Where-Object { $_ -le $detected } | Select-Object -Last 1)
    if ($best.Count -gt 0) {
        Write-Output "compute_$($best[0])"
        exit 0
    }
}

$fallbackNumber = Get-ArchNumber $fallback
if ($supported.Count -gt 0 -and $fallbackNumber -gt 0 -and -not ($supported -contains $fallbackNumber)) {
    $bestFallback = @($supported | Select-Object -Last 1)
    if ($bestFallback.Count -gt 0) {
        Write-Output "compute_$($bestFallback[0])"
        exit 0
    }
}

Write-Output $fallback
