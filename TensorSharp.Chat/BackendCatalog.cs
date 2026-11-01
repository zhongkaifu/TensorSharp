// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;

namespace TensorSharp.Server
{
    public sealed record BackendOption(string Value, string Label);

    /// <summary>
    /// The backend vocabulary: the ordered descriptor table every host offers, the
    /// canonical spelling of each backend name and its <see cref="BackendType"/>
    /// mapping.
    ///
    /// <para>
    /// This is the pure half. Deciding which of these backends is actually usable on
    /// the current machine means touching the CUDA, MLX and GGML backends, and those
    /// assemblies are not something a host-neutral library can pull in (an iOS
    /// consumer cannot link TensorSharp.Backends.Cuda at all). So the probes are handed
    /// in as delegates: TensorSharp.Server passes its real ones from
    /// <c>BackendCatalogProbes</c>, and an app that knows its one backend passes
    /// constants.
    /// </para>
    /// </summary>
    public static class BackendCatalog
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

        /// <summary>
        /// The descriptor table filtered by the host's availability probes, in UI order.
        /// Every probe is required: a missing one used to fall back to the real backend
        /// probe, which is exactly the dependency this library exists to avoid.
        /// </summary>
        internal static IReadOnlyList<BackendOption> GetSupportedBackends(
            Func<GgmlBackendType, bool> isGgmlBackendAvailable,
            Func<bool> isCudaBackendAvailable,
            Func<bool> isMlxBackendAvailable)
        {
            if (isGgmlBackendAvailable == null) throw new ArgumentNullException(nameof(isGgmlBackendAvailable));
            if (isCudaBackendAvailable == null) throw new ArgumentNullException(nameof(isCudaBackendAvailable));
            if (isMlxBackendAvailable == null) throw new ArgumentNullException(nameof(isMlxBackendAvailable));

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

        public static string? Canonicalize(string? backend)
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

        internal static bool IsKnownBackend(string? backend)
        {
            string? canonical = Canonicalize(backend);
            return BackendDescriptors.Any(descriptor => descriptor.Value == canonical);
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

        private sealed record BackendDescriptor(string Value, string Label, GgmlBackendType? GgmlBackendType, bool AlwaysAvailable);
    }
}
