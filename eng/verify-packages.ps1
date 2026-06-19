param(
    [string] $Configuration = "Release",
    [string] $OutputDirectory = "artifacts/packages",
    # Overrides the package version baked into each .nupkg. Empty = use the
    # version from Directory.Build.props ($(TensorSharpVersion)). The publish
    # pipeline passes the git tag here (e.g. 2.8.6 from tag v2.8.6).
    [string] $PackageVersion = "",
    # Skip the native (CMake/CUDA/MLX) build steps while packing. The produced
    # packages contain only managed assemblies, so the native libraries are not
    # required to pack them — this lets packing run on a host without the native
    # toolchain.
    [switch] $SkipNativeBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $PackageOutput = $OutputDirectory
}
else {
    $PackageOutput = Join-Path $RepoRoot $OutputDirectory
}

$PublicPackages = @(
    @{
        Id = "TensorSharp.Core"
        Project = "TensorSharp.Core/TensorSharp.Core.csproj"
        TensorSharpDependencies = @()
        EmbeddedAssemblies = @("AdvUtils.dll")
    },
    @{
        Id = "TensorSharp.Runtime"
        Project = "TensorSharp.Runtime/TensorSharp.Runtime.csproj"
        TensorSharpDependencies = @()
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Backends.Cuda"
        Project = "TensorSharp.Backends.Cuda/TensorSharp.Backends.Cuda.csproj"
        TensorSharpDependencies = @("TensorSharp.Core")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Backends.GGML"
        Project = "TensorSharp.Backends.GGML/TensorSharp.Backends.GGML.csproj"
        TensorSharpDependencies = @("TensorSharp.Core")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Backends.MLX"
        Project = "TensorSharp.Backends.MLX/TensorSharp.Backends.MLX.csproj"
        TensorSharpDependencies = @("TensorSharp.Core", "TensorSharp.Runtime")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Models"
        Project = "TensorSharp.Models/TensorSharp.Models.csproj"
        TensorSharpDependencies = @("TensorSharp.Core", "TensorSharp.Runtime", "TensorSharp.Backends.GGML", "TensorSharp.Backends.Cuda", "TensorSharp.Backends.MLX")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Server"
        Project = "TensorSharp.Server/TensorSharp.Server.csproj"
        TensorSharpDependencies = @("TensorSharp.Runtime", "TensorSharp.Models", "TensorSharp.Backends.GGML", "TensorSharp.Backends.Cuda", "TensorSharp.Backends.MLX")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Cli"
        Project = "TensorSharp.Cli/TensorSharp.Cli.csproj"
        TensorSharpDependencies = @("TensorSharp.Core", "TensorSharp.Runtime", "TensorSharp.Models", "TensorSharp.Backends.GGML", "TensorSharp.Backends.MLX")
        EmbeddedAssemblies = @()
    }
)

function Invoke-CheckedDotNet {
    param([string[]] $Arguments)

    & dotnet @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Get-AssemblySafeVersion {
    param([string] $Version)

    # AssemblyVersion/FileVersion are System.Version values: at most four parts,
    # each a UInt16 (0-65535). Return the numeric core when it fits, otherwise
    # $null so the caller can leave the assembly version at the repo default.
    # Strip any SemVer prerelease/build-metadata suffix (from the first '-' or
    # '+') because the assembly version only understands the numeric core.
    $core = ($Version -split '[-+]', 2)[0]
    $parts = $core.Split('.')
    if ($parts.Count -lt 1 -or $parts.Count -gt 4) {
        return $null
    }
    foreach ($part in $parts) {
        $n = 0
        if (-not [int]::TryParse($part, [ref] $n) -or $n -lt 0 -or $n -gt 65535) {
            return $null
        }
    }
    return $core
}

function Get-NormalizedNuGetVersion {
    param([string] $Version)

    # Replicate NuGetVersion.ToNormalizedString for the version segment of the
    # .nupkg file name: drop build metadata (+...), split the numeric core from
    # the prerelease label (-...), pad the core to at least Major.Minor.Patch,
    # drop a trailing zero revision, and strip leading zeros from each segment.
    # e.g. "20260618" -> "20260618.0.0", "2.8.6.0" -> "2.8.6", "2.8.6" -> "2.8.6".
    $noMetadata = ($Version -split '\+', 2)[0]
    $coreAndPre = $noMetadata -split '-', 2
    $core = $coreAndPre[0]
    $prerelease = if ($coreAndPre.Count -gt 1) { $coreAndPre[1] } else { $null }

    $nums = @($core.Split('.') | ForEach-Object { [int]::Parse($_, [System.Globalization.CultureInfo]::InvariantCulture) })
    while ($nums.Count -lt 3) { $nums += 0 }
    if ($nums.Count -eq 4 -and $nums[3] -eq 0) { $nums = @($nums[0..2]) }

    $normalized = ($nums -join '.')
    if (-not [string]::IsNullOrEmpty($prerelease)) {
        $normalized += "-$prerelease"
    }
    return $normalized
}

function Get-ProjectPackageProperties {
    param([string] $ProjectPath)

    $json = & dotnet msbuild $ProjectPath -getProperty:PackageId -getProperty:PackageVersion -getProperty:Version -getProperty:IsPackable
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet msbuild property query failed for $ProjectPath"
    }

    return ($json -join "`n") | ConvertFrom-Json
}

function Read-Nupkg {
    param([string] $PackagePath)

    Add-Type -AssemblyName System.IO.Compression.FileSystem

    $zip = [System.IO.Compression.ZipFile]::OpenRead($PackagePath)
    try {
        $entries = @($zip.Entries | ForEach-Object { $_.FullName })
        $nuspecEntry = $zip.Entries | Where-Object { $_.FullName -like "*.nuspec" } | Select-Object -First 1
        if ($null -eq $nuspecEntry) {
            throw "No .nuspec entry found in $PackagePath"
        }

        $stream = $nuspecEntry.Open()
        try {
            $reader = [System.IO.StreamReader]::new($stream)
            try {
                [xml] $nuspec = $reader.ReadToEnd()
            }
            finally {
                $reader.Dispose()
            }
        }
        finally {
            $stream.Dispose()
        }

        $dependencyNodes = $nuspec.SelectNodes("//*[local-name()='dependency']")
        $dependencies = @(
            foreach ($dependency in $dependencyNodes) {
                [pscustomobject] @{
                    Id = $dependency.id
                    Version = $dependency.version
                }
            }
        )

        return [pscustomobject] @{
            Entries = $entries
            Dependencies = $dependencies
        }
    }
    finally {
        $zip.Dispose()
    }
}

function Assert-SameSet {
    param(
        [string] $PackageId,
        [string[]] $Actual,
        [string[]] $Expected
    )

    $actualSet = @($Actual | Sort-Object -Unique)
    $expectedSet = @($Expected | Sort-Object -Unique)
    $unexpected = @($actualSet | Where-Object { $expectedSet -notcontains $_ })
    $missing = @($expectedSet | Where-Object { $actualSet -notcontains $_ })

    if ($unexpected.Count -gt 0 -or $missing.Count -gt 0) {
        $actualText = if ($actualSet.Count -eq 0) { "(none)" } else { $actualSet -join ", " }
        $expectedText = if ($expectedSet.Count -eq 0) { "(none)" } else { $expectedSet -join ", " }
        throw "$PackageId has unexpected TensorSharp package dependencies. Expected: $expectedText. Actual: $actualText."
    }
}

New-Item -ItemType Directory -Force -Path $PackageOutput | Out-Null

# Capture the override into a distinctly-named variable: PowerShell variable
# names are case-insensitive, so reusing $PackageVersion inside the loop (where
# $packageVersion holds each project's resolved version) would clobber it.
$RequestedVersion = $PackageVersion

# Extra MSBuild properties applied to every `dotnet pack` invocation below.
$ExtraPackArgs = @()
if (-not [string]::IsNullOrWhiteSpace($RequestedVersion)) {
    # The NuGet package version accepts the full requested string (large or
    # date-style versions like 20260618 are valid here).
    $ExtraPackArgs += "-p:PackageVersion=$RequestedVersion"

    # Line AssemblyVersion/FileVersion up with the package version when it fits
    # the System.Version range; setting Version drives both. A date-style or
    # otherwise out-of-range tag would make the C# compiler fail with CS7034, so
    # in that case leave the assemblies at the repo default ($(TensorSharpVersion))
    # and just record the true version in the (free-form) informational version.
    $assemblySafeVersion = Get-AssemblySafeVersion $RequestedVersion
    if ($null -ne $assemblySafeVersion) {
        $ExtraPackArgs += "-p:Version=$assemblySafeVersion"
    }
    else {
        $ExtraPackArgs += "-p:InformationalVersion=$RequestedVersion"
        Write-Warning "Requested version '$RequestedVersion' is not a valid assembly version (each part must be 0-65535); assemblies keep the repo default while the package version is '$RequestedVersion'."
    }
}
if ($SkipNativeBuild) {
    $ExtraPackArgs += "-p:TensorSharpSkipGgmlNative=true"
    $ExtraPackArgs += "-p:TensorSharpSkipMlxNative=true"
}

foreach ($package in $PublicPackages) {
    $projectPath = Join-Path $RepoRoot $package.Project
    $properties = Get-ProjectPackageProperties $projectPath
    $packageId = $properties.Properties.PackageId
    $packageVersion = $properties.Properties.PackageVersion
    if ([string]::IsNullOrWhiteSpace($packageVersion)) {
        $packageVersion = $properties.Properties.Version
    }
    # The msbuild query above reports the project's default version; when the
    # caller overrides it the produced .nupkg is named with the override.
    if (-not [string]::IsNullOrWhiteSpace($RequestedVersion)) {
        $packageVersion = $RequestedVersion
    }

    if ($packageId -ne $package.Id) {
        throw "$projectPath has PackageId '$packageId'; expected '$($package.Id)'."
    }

    if ($properties.Properties.IsPackable -ne "true") {
        throw "$packageId must be packable because it is listed in README.md."
    }

    Invoke-CheckedDotNet (@("pack", $projectPath, "-c", $Configuration, "-o", $PackageOutput) + $ExtraPackArgs)

    # NuGet normalizes the version in the produced file name (e.g. a bare
    # "20260618" becomes "20260618.0.0"), so match against the normalized form.
    $normalizedVersion = Get-NormalizedNuGetVersion $packageVersion
    $nupkgPath = Join-Path $PackageOutput "$packageId.$normalizedVersion.nupkg"
    if (-not (Test-Path $nupkgPath)) {
        throw "Expected package was not created: $nupkgPath"
    }

    $nupkg = Read-Nupkg $nupkgPath
    $internalDependencies = @(
        $nupkg.Dependencies |
            Where-Object { $_.Id -eq "AdvUtils" -or $_.Id -like "TensorSharp.*" } |
            ForEach-Object { $_.Id }
    )

    Assert-SameSet -PackageId $packageId -Actual $internalDependencies -Expected $package.TensorSharpDependencies

    foreach ($assemblyName in $package.EmbeddedAssemblies) {
        $assemblyEntry = $nupkg.Entries | Where-Object { $_ -like "lib/*/$assemblyName" } | Select-Object -First 1
        if ($null -eq $assemblyEntry) {
            throw "$packageId must embed $assemblyName because it is an internal implementation dependency."
        }
    }

    Write-Host "Verified $packageId $packageVersion"
}

Write-Host "Package verification succeeded. Output: $PackageOutput"
