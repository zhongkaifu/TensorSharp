param(
    [string] $Configuration = "Release",
    [string] $OutputDirectory = "artifacts/packages"
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
        Id = "TensorSharp.Models"
        Project = "TensorSharp.Models/TensorSharp.Models.csproj"
        TensorSharpDependencies = @("TensorSharp.Core", "TensorSharp.Runtime", "TensorSharp.Backends.GGML", "TensorSharp.Backends.Cuda")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Server"
        Project = "TensorSharp.Server/TensorSharp.Server.csproj"
        TensorSharpDependencies = @("TensorSharp.Runtime", "TensorSharp.Models", "TensorSharp.Backends.GGML", "TensorSharp.Backends.Cuda")
        EmbeddedAssemblies = @()
    },
    @{
        Id = "TensorSharp.Cli"
        Project = "TensorSharp.Cli/TensorSharp.Cli.csproj"
        TensorSharpDependencies = @("TensorSharp.Core", "TensorSharp.Runtime", "TensorSharp.Models", "TensorSharp.Backends.GGML")
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

foreach ($package in $PublicPackages) {
    $projectPath = Join-Path $RepoRoot $package.Project
    $properties = Get-ProjectPackageProperties $projectPath
    $packageId = $properties.Properties.PackageId
    $packageVersion = $properties.Properties.PackageVersion
    if ([string]::IsNullOrWhiteSpace($packageVersion)) {
        $packageVersion = $properties.Properties.Version
    }

    if ($packageId -ne $package.Id) {
        throw "$projectPath has PackageId '$packageId'; expected '$($package.Id)'."
    }

    if ($properties.Properties.IsPackable -ne "true") {
        throw "$packageId must be packable because it is listed in README.md."
    }

    Invoke-CheckedDotNet @("pack", $projectPath, "-c", $Configuration, "-o", $PackageOutput)

    $nupkgPath = Join-Path $PackageOutput "$packageId.$packageVersion.nupkg"
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
