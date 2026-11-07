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
        # Package id intentionally differs from the project name - see the
        # PackageId comment in TensorSharp.Core.csproj.
        Id = "TensorSharp.Tensors"
        Project = "TensorSharp.Core/TensorSharp.Core.csproj"
        TensorSharpDependencies = @()
        EmbeddedAssemblies = @("AdvUtils.dll")
    },
    @{
        Id = "TensorSharp.Runtime.Logging"
        Project = "TensorSharp.Runtime.Logging/TensorSharp.Runtime.Logging.csproj"
        TensorSharpDependencies = @()
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Runtime"
        Project = "TensorSharp.Runtime/TensorSharp.Runtime.csproj"
        TensorSharpDependencies = @()
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Knowledge"
        Project = "TensorSharp.Knowledge/TensorSharp.Knowledge.csproj"
        TensorSharpDependencies = @()
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.AgentHost"
        Project = "TensorSharp.AgentHost/TensorSharp.AgentHost.csproj"
        TensorSharpDependencies = @("TensorSharp.Runtime", "TensorSharp.Knowledge")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Backends.Cuda"
        Project = "TensorSharp.Backends.Cuda/TensorSharp.Backends.Cuda.csproj"
        TensorSharpDependencies = @("TensorSharp.Tensors")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Backends.GGML"
        Project = "TensorSharp.Backends.GGML/TensorSharp.Backends.GGML.csproj"
        TensorSharpDependencies = @("TensorSharp.Tensors")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Backends.MLX"
        Project = "TensorSharp.Backends.MLX/TensorSharp.Backends.MLX.csproj"
        TensorSharpDependencies = @("TensorSharp.Tensors", "TensorSharp.Runtime")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Distributed"
        Project = "TensorSharp.Distributed/TensorSharp.Distributed.csproj"
        TensorSharpDependencies = @("TensorSharp.Tensors", "TensorSharp.Backends.Cuda")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Models"
        Project = "TensorSharp.Models/TensorSharp.Models.csproj"
        TensorSharpDependencies = @("TensorSharp.Tensors", "TensorSharp.Runtime", "TensorSharp.Backends.GGML", "TensorSharp.Backends.Cuda", "TensorSharp.Backends.MLX")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Studio"
        Project = "TensorSharp.Studio/TensorSharp.Studio.csproj"
        TensorSharpDependencies = @()
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Chat"
        Project = "TensorSharp.Chat/TensorSharp.Chat.csproj"
        TensorSharpDependencies = @("TensorSharp.Runtime", "TensorSharp.Runtime.Logging", "TensorSharp.AgentHost", "TensorSharp.Models", "TensorSharp.Backends.GGML", "TensorSharp.Knowledge", "TensorSharp.Studio")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Server"
        Project = "TensorSharp.Server/TensorSharp.Server.csproj"
        TensorSharpDependencies = @("TensorSharp.Runtime", "TensorSharp.Runtime.Logging", "TensorSharp.AgentHost", "TensorSharp.Chat", "TensorSharp.Models", "TensorSharp.Backends.GGML", "TensorSharp.Backends.Cuda", "TensorSharp.Backends.MLX", "TensorSharp.Distributed", "TensorSharp.Knowledge")
        EmbeddedAssemblies = @()
    },
    @{
        # The opt-in OpenTelemetry pipeline. Host-only: only the two hosts
        # below reference it, so a consumer of the libraries never gets the
        # OpenTelemetry SDK.
        Id = "TensorSharp.Telemetry"
        Project = "TensorSharp.Telemetry/TensorSharp.Telemetry.csproj"
        TensorSharpDependencies = @("TensorSharp.Runtime")
        EmbeddedAssemblies = @()
    },
    @{
        # The runnable server. TensorSharp.Server is the library it builds on;
        # TensorSharp.Telemetry is its opt-in OpenTelemetry pipeline.
        Id = "TensorSharp.Server.Host"
        Project = "TensorSharp.Server.Host/TensorSharp.Server.Host.csproj"
        TensorSharpDependencies = @("TensorSharp.Server", "TensorSharp.Telemetry")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Cli"
        Project = "TensorSharp.Cli/TensorSharp.Cli.csproj"
        TensorSharpDependencies = @("TensorSharp.Tensors", "TensorSharp.Runtime", "TensorSharp.Runtime.Logging", "TensorSharp.AgentHost", "TensorSharp.Chat", "TensorSharp.Models", "TensorSharp.Backends.GGML", "TensorSharp.Backends.MLX", "TensorSharp.Distributed", "TensorSharp.Telemetry")
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
            Id = [string] $nuspec.package.metadata.id
            Version = [string] $nuspec.package.metadata.version
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

function Assert-PackageManifest {
    param([object[]] $Packages)

    $ids = @($Packages | ForEach-Object { $_.Id })
    if (@($ids | Sort-Object -Unique).Count -ne $ids.Count) {
        throw "Public package manifest contains duplicate package IDs."
    }
    foreach ($package in $Packages) {
        foreach ($dependencyId in $package.TensorSharpDependencies) {
            if ($ids -notcontains $dependencyId) {
                throw "$($package.Id) depends on $dependencyId, which is missing from the public package manifest."
            }
        }
    }
}

function Assert-PackageClosure {
    param([hashtable] $PackagesById)

    foreach ($package in $PackagesById.Values) {
        if (-not (Test-Path -LiteralPath $package.Path -PathType Leaf)) {
            throw "Missing package artifact for $($package.Id): $($package.Path)"
        }
        foreach ($dependency in $package.Dependencies) {
            if ($dependency.Id -ne "AdvUtils" -and $dependency.Id -notlike "TensorSharp.*") {
                continue
            }
            if (-not $PackagesById.ContainsKey($dependency.Id)) {
                throw "$($package.Id) depends on $($dependency.Id), which has no verified package artifact."
            }
            $targetVersion = $PackagesById[$dependency.Id].Version
            # SDK pack emits a minimum version; also accept its explicit range
            # and exact-version forms. Every internal edge must name the version
            # built in this run, rather than silently resolving an older release.
            if ($dependency.Version -notin @($targetVersion, "[$targetVersion, )", "[$targetVersion]")) {
                throw "$($package.Id) dependency $($dependency.Id) has version '$($dependency.Version)'; expected '$targetVersion'."
            }
        }
    }
}

Assert-PackageManifest -Packages $PublicPackages
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

$VerifiedPackages = @{}
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
        throw "$packageId must be packable because this script publishes it."
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
    $actualVersion = Get-NormalizedNuGetVersion $nupkg.Version
    if ($nupkg.Id -ne $packageId -or $actualVersion -ne $normalizedVersion) {
        throw "$nupkgPath has unexpected package identity '$($nupkg.Id) $($nupkg.Version)'; expected '$packageId $normalizedVersion'."
    }
    $internalDependencies = @(
        $nupkg.Dependencies |
            Where-Object { $_.Id -eq "AdvUtils" -or $_.Id -like "TensorSharp.*" } |
            ForEach-Object { $_.Id }
    )

    Assert-SameSet -PackageId $packageId -Actual $internalDependencies -Expected $package.TensorSharpDependencies

    $VerifiedPackages[$packageId] = [pscustomobject] @{
        Path = $nupkgPath
        Id = $nupkg.Id
        Version = $actualVersion
        Dependencies = $nupkg.Dependencies
    }

    foreach ($assemblyName in $package.EmbeddedAssemblies) {
        $assemblyEntry = $nupkg.Entries | Where-Object { $_ -like "lib/*/$assemblyName" } | Select-Object -First 1
        if ($null -eq $assemblyEntry) {
            throw "$packageId must embed $assemblyName because it is an internal implementation dependency."
        }
    }

    Write-Host "Verified $packageId $packageVersion"
}

Assert-PackageClosure -PackagesById $VerifiedPackages
Write-Host "Package verification succeeded. Output: $PackageOutput"
