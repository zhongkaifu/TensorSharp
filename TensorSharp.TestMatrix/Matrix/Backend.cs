// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

using System.Runtime.InteropServices;

namespace TensorSharp.TestMatrix.Matrix;

/// <summary>
/// One TensorSharp compute backend. Mirrors <c>TensorSharp.Server.BackendCatalog</c>
/// (cpu, ggml_cpu, ggml_cuda, ggml_metal, cuda, mlx). Kept as a flat record here so
/// the test runner stays decoupled from internal TensorSharp APIs.
/// </summary>
public sealed record BackendInfo(
    string Id,
    string DisplayName,
    bool RequiresCuda,
    bool RequiresMetal,
    bool RequiresAppleSilicon,
    bool RequiresVulkan = false)
{
    public bool IsAvailableOnHost()
    {
        if (RequiresAppleSilicon && (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
            || RuntimeInformation.OSArchitecture != Architecture.Arm64))
        {
            return false;
        }
        if (RequiresMetal && !RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return false;
        }
        if (RequiresCuda)
        {
            // Heuristic: assume CUDA is available on non-OSX hosts when not explicitly
            // disabled. The actual check happens when TensorSharp.Cli starts; if CUDA
            // is missing the runner records a 'backend unavailable' error.
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                return false;
            }
        }
        if (RequiresVulkan)
        {
            // Same heuristic as CUDA: ggml-vulkan is built for Windows/Linux only
            // (Metal is the GPU backend on macOS); the real device check happens
            // when TensorSharp.Cli starts.
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                return false;
            }
        }
        return true;
    }
}

public static class BackendCatalog
{
    public static readonly BackendInfo Cpu = new(
        Id: "cpu",
        DisplayName: "CPU (Pure C#)",
        RequiresCuda: false,
        RequiresMetal: false,
        RequiresAppleSilicon: false);

    public static readonly BackendInfo GgmlCpu = new(
        Id: "ggml_cpu",
        DisplayName: "GGML CPU",
        RequiresCuda: false,
        RequiresMetal: false,
        RequiresAppleSilicon: false);

    public static readonly BackendInfo GgmlMetal = new(
        Id: "ggml_metal",
        DisplayName: "GGML Metal (GPU)",
        RequiresCuda: false,
        RequiresMetal: true,
        RequiresAppleSilicon: false);

    public static readonly BackendInfo GgmlCuda = new(
        Id: "ggml_cuda",
        DisplayName: "GGML CUDA (GPU)",
        RequiresCuda: true,
        RequiresMetal: false,
        RequiresAppleSilicon: false);

    public static readonly BackendInfo GgmlVulkan = new(
        Id: "ggml_vulkan",
        DisplayName: "GGML Vulkan (GPU)",
        RequiresCuda: false,
        RequiresMetal: false,
        RequiresAppleSilicon: false,
        RequiresVulkan: true);

    public static readonly BackendInfo DirectCuda = new(
        Id: "cuda",
        DisplayName: "CUDA (cuBLAS GPU)",
        RequiresCuda: true,
        RequiresMetal: false,
        RequiresAppleSilicon: false);

    public static readonly BackendInfo Mlx = new(
        Id: "mlx",
        DisplayName: "MLX Metal (GPU)",
        RequiresCuda: false,
        RequiresMetal: true,
        RequiresAppleSilicon: true);

    public static readonly IReadOnlyList<BackendInfo> All = new[]
    {
        Cpu, GgmlCpu, GgmlMetal, GgmlCuda, GgmlVulkan, DirectCuda, Mlx,
    };

    public static BackendInfo? FindById(string id)
    {
        foreach (BackendInfo b in All)
        {
            if (string.Equals(b.Id, id, StringComparison.OrdinalIgnoreCase))
            {
                return b;
            }
        }
        return null;
    }
}
