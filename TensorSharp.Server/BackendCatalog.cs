using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Server
{
    internal sealed record BackendOption(string Value, string Label);

    internal static class BackendCatalog
    {
        // TensorSharp.Server should always expose the two CPU choices distinctly:
        // `ggml_cpu` is the native GGML CPU backend, while `cpu` is the pure C# backend.
        private static readonly BackendDescriptor[] BackendDescriptors =
        {
            new("mlx", "MLX Metal (GPU)", null, AlwaysAvailable: false),
            new("cuda", "CUDA (cuBLAS GPU)", null, AlwaysAvailable: false),
            new("ggml_metal", "GGML Metal (GPU)", GgmlBackendType.Metal, AlwaysAvailable: false),
            new("ggml_cuda", "GGML CUDA (GPU)", GgmlBackendType.Cuda, AlwaysAvailable: false),
            new("ggml_vulkan", "GGML Vulkan (GPU)", GgmlBackendType.Vulkan, AlwaysAvailable: false),
            new("ggml_cpu", "GGML CPU", GgmlBackendType.Cpu, AlwaysAvailable: true),
            new("cpu", "CPU (Pure C#)", GgmlBackendType.Cpu, AlwaysAvailable: true),
        };

        internal static IReadOnlyList<BackendOption> GetSupportedBackends(
            Func<GgmlBackendType, bool> isGgmlBackendAvailable = null,
            Func<bool> isCudaBackendAvailable = null,
            Func<bool> isMlxBackendAvailable = null)
        {
            isGgmlBackendAvailable ??= IsGgmlBackendAvailable;
            isCudaBackendAvailable ??= CudaBackend.IsAvailable;
            isMlxBackendAvailable ??= MlxBackend.IsAvailable;

            return BackendDescriptors
                .Where(descriptor => descriptor.AlwaysAvailable ||
                    (string.Equals(descriptor.Value, "mlx", StringComparison.OrdinalIgnoreCase)
                        ? isMlxBackendAvailable()
                        : descriptor.GgmlBackendType.HasValue
                        ? isGgmlBackendAvailable(descriptor.GgmlBackendType.Value)
                        : isCudaBackendAvailable()))
                .Select(descriptor => new BackendOption(descriptor.Value, descriptor.Label))
                .ToArray();
        }

        internal static string ResolveDefaultBackend(string configuredBackend, IReadOnlyList<BackendOption> supportedBackends)
        {
            string canonicalBackend = Canonicalize(configuredBackend);
            if (!string.IsNullOrEmpty(canonicalBackend) &&
                supportedBackends.Any(backend => string.Equals(backend.Value, canonicalBackend, StringComparison.OrdinalIgnoreCase)))
            {
                return canonicalBackend;
            }

            return supportedBackends.FirstOrDefault()?.Value ?? canonicalBackend ?? configuredBackend;
        }

        internal static string Canonicalize(string backend)
        {
            if (string.IsNullOrWhiteSpace(backend))
                return null;

            return backend.Trim().ToLowerInvariant() switch
            {
                "mlx" or "mlx_metal" or "mlx-metal" => "mlx",
                "cuda" or "direct_cuda" or "direct-cuda" => "cuda",
                "ggml_cuda" or "ggml-cuda" => "ggml_cuda",
                "ggml_vulkan" or "ggml-vulkan" => "ggml_vulkan",
                "ggml_metal" => "ggml_metal",
                "ggml_cpu" => "ggml_cpu",
                "cpu" => "cpu",
                var value => value,
            };
        }

        internal static string ToBackendValue(BackendType backendType)
        {
            return backendType switch
            {
                BackendType.Mlx => "mlx",
                BackendType.Cuda => "cuda",
                BackendType.GgmlMetal => "ggml_metal",
                BackendType.GgmlCuda => "ggml_cuda",
                BackendType.GgmlVulkan => "ggml_vulkan",
                BackendType.GgmlCpu => "ggml_cpu",
                BackendType.Cpu => "cpu",
                _ => null,
            };
        }

        private static bool IsGgmlBackendAvailable(GgmlBackendType backendType)
        {
            try
            {
                // Backend discovery runs at web-app startup, so it must not spin up
                // any GGML device — otherwise picking a non-GGML backend (MLX,
                // direct CUDA) would still trigger `ggml_metal_device_init` / etc.
                // logs at startup. CanInitializeBackend is a lightweight compile-flag
                // + platform check; the real GGML init is deferred until a GGML
                // backend is actually selected.
                return GgmlBasicOps.CanInitializeBackend(backendType);
            }
            catch
            {
                return false;
            }
        }

        private sealed record BackendDescriptor(string Value, string Label, GgmlBackendType? GgmlBackendType, bool AlwaysAvailable);
    }
}



