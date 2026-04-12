using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;

namespace TensorSharp.Server
{
    internal sealed record BackendOption(string Value, string Label);

    internal static class BackendCatalog
    {
        // TensorSharp.Server should always expose the two CPU choices distinctly:
        // `ggml_cpu` is the native GGML CPU backend, while `cpu` is the pure C# backend.
        private static readonly BackendDescriptor[] BackendDescriptors =
        {
            new("ggml_metal", "GGML Metal (GPU)", GgmlBackendType.Metal, AlwaysAvailable: false),
            new("ggml_cuda", "GGML CUDA (GPU)", GgmlBackendType.Cuda, AlwaysAvailable: false),
            new("ggml_cpu", "GGML CPU", GgmlBackendType.Cpu, AlwaysAvailable: true),
            new("cpu", "CPU (Pure C#)", GgmlBackendType.Cpu, AlwaysAvailable: true),
        };

        internal static IReadOnlyList<BackendOption> GetSupportedBackends(Func<GgmlBackendType, bool> isGgmlBackendAvailable = null)
        {
            isGgmlBackendAvailable ??= IsGgmlBackendAvailable;

            return BackendDescriptors
                .Where(descriptor => descriptor.AlwaysAvailable || isGgmlBackendAvailable(descriptor.GgmlBackendType))
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
                "cuda" or "ggml_cuda" => "ggml_cuda",
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
                BackendType.GgmlMetal => "ggml_metal",
                BackendType.GgmlCuda => "ggml_cuda",
                BackendType.GgmlCpu => "ggml_cpu",
                BackendType.Cpu => "cpu",
                _ => null,
            };
        }

        private static bool IsGgmlBackendAvailable(GgmlBackendType backendType)
        {
            try
            {
                // Backend discovery runs at web-app startup, so it must not claim the
                // process-wide GGML backend before the user actually loads a model.
                return GgmlBasicOps.CanInitializeBackend(backendType);
            }
            catch
            {
                return false;
            }
        }

        private sealed record BackendDescriptor(string Value, string Label, GgmlBackendType GgmlBackendType, bool AlwaysAvailable);
    }
}



