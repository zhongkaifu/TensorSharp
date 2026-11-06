#!/usr/bin/env bash
# Runs InferenceWeb.Tests/MappedTableWarmTests.cs on Linux (WSL on a Windows box, or any Linux host) without
# building the real test project, which drags in the CUDA/GGML/MLX native builds: the Linux-only page-cache tests
# (mincore re-warm, posix_fadvise drop, MADV_RANDOM) are skipped on Windows, and this is how they get run there.
#
# The file under test and its two sources (TensorSharp.Models/PageCacheHints.cs, MappedTableWarm.cs) use only the
# BCL, so a throwaway xunit project in $HOME compiles them together. It is built under $HOME, never in the repo:
# the repo's bin/ folders are shared with the Windows build.
#
# Usage: eng/validation/wsl-host-page-cache-tests.sh [repo-root] [extra dotnet test args...]
# NUGET_PACKAGES defaults to the Windows user's package cache under /mnt/c when it exists (restore works offline
# from it); set it to use another.
set -euo pipefail

repo="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
shift || true
work="${TS_HOST_PAGE_CACHE_TEST_DIR:-$HOME/ts-host-page-cache-tests}"

if [[ -z "${NUGET_PACKAGES:-}" ]]; then
  for candidate in /mnt/c/Users/*/.nuget/packages; do
    if [[ -d "$candidate/xunit" ]]; then
      export NUGET_PACKAGES="$candidate"
      break
    fi
  done
fi

rm -rf "$work"
mkdir -p "$work"
cp "$repo/TensorSharp.Models/PageCacheHints.cs" "$repo/TensorSharp.Models/MappedTableWarm.cs" \
   "$repo/InferenceWeb.Tests/MappedTableWarmTests.cs" "$work/"

cat > "$work/HostPageCacheTests.csproj" <<'XML'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>annotations</Nullable>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
    <IsPackable>false</IsPackable>
    <NuGetAudit>false</NuGetAudit>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.14.1" />
    <PackageReference Include="xunit" Version="2.9.3" />
    <PackageReference Include="xunit.runner.visualstudio" Version="3.1.4" />
  </ItemGroup>
</Project>
XML
printf 'global using Xunit;\n' > "$work/GlobalUsings.cs"

echo "[host-page-cache-tests] sources from $repo"
sha256sum "$work"/*.cs
cd "$work"
dotnet test HostPageCacheTests.csproj -c Release --logger "console;verbosity=normal" "$@"
